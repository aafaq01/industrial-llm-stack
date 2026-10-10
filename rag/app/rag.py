"""Haystack pipelines: indexing, hybrid retrieval with reranking, and generation via llama.cpp."""

import os
import re
from collections.abc import Callable

from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchBM25Retriever,
    ElasticsearchEmbeddingRetriever,
)
from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)

ES_URL = os.environ.get("ES_URL", "http://elasticsearch:9200")
ES_INDEX = os.environ.get("ES_INDEX", "corpus")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llamacpp:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-1.5b-instruct-q4_k_m")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "384"))
RETRIEVE_K = 10  # per retriever, before fusion
TOP_K = 5  # after reranking, goes into the prompt
# bge v1.5 recommends this instruction for retrieval queries.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Repeated NIST page footer; adds noise to every chunk.
FOOTER = r"This publication is available free of charge from:?\s*https?://\S+"

CITE_REMINDER = "\n\nAnswer in one to three sentences. End each sentence with the source number(s) it came from, like [2]."
SYSTEM_PROMPT = """You answer questions about engineering documents using only the numbered sources below.
Cite every claim with the source number in square brackets, for example [1] or [2][3].
If the sources do not contain the answer, say you do not know. Be concise.

Sources:
{sources}"""


def document_store() -> ElasticsearchDocumentStore:
    return ElasticsearchDocumentStore(
        hosts=ES_URL,
        index=ES_INDEX,
        basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
    )


def cleaner() -> DocumentCleaner:
    return DocumentCleaner(remove_regex=FOOTER)


def splitter() -> DocumentSplitter:
    # ponytail: fixed 150-word windows with 30 overlap; sentence-aware splitting if answers get cut mid-thought.
    # Small chunks because llama.cpp runs --ctx-size 4096 --parallel 2, so each request gets 2048 tokens.
    return DocumentSplitter(split_by="word", split_length=150, split_overlap=30)


def chunker() -> Pipeline:
    """Cleaning + chunking only, so it can be tested without models or Elasticsearch."""
    p = Pipeline()
    p.add_component("clean", cleaner())
    p.add_component("split", splitter())
    p.connect("clean", "split")
    return p


def indexing_pipeline(store: ElasticsearchDocumentStore) -> Pipeline:
    p = Pipeline()
    p.add_component("convert", PyPDFToDocument())
    p.add_component("clean", cleaner())
    p.add_component("split", splitter())
    p.add_component(
        "embed", SentenceTransformersDocumentEmbedder(model=EMBED_MODEL, batch_size=64)
    )
    p.add_component("write", DocumentWriter(store, policy=DuplicatePolicy.OVERWRITE))
    p.connect("convert", "clean")
    p.connect("clean", "split")
    p.connect("split", "embed")
    p.connect("embed", "write")
    return p


def retrieval_pipeline(store: ElasticsearchDocumentStore) -> Pipeline:
    p = Pipeline()
    p.add_component(
        "embed_query",
        SentenceTransformersTextEmbedder(model=EMBED_MODEL, prefix=QUERY_PREFIX),
    )
    p.add_component(
        "bm25_retrieve",
        ElasticsearchBM25Retriever(document_store=store, top_k=RETRIEVE_K),
    )
    p.add_component(
        "dense_retrieve",
        ElasticsearchEmbeddingRetriever(document_store=store, top_k=RETRIEVE_K),
    )
    p.add_component("join", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
    p.add_component(
        "rerank", SentenceTransformersSimilarityRanker(model=RERANK_MODEL, top_k=TOP_K)
    )
    p.connect("embed_query.embedding", "dense_retrieve.query_embedding")
    p.connect("bm25_retrieve", "join")
    p.connect("dense_retrieve", "join")
    p.connect("join", "rerank")
    return p


def generation_pipeline() -> Pipeline:
    p = Pipeline()
    p.add_component(
        "generate",
        OpenAIChatGenerator(
            api_key=Secret.from_token("not-needed"),  # llama.cpp ignores it
            model=LLM_MODEL,
            api_base_url=LLM_BASE_URL,
            generation_kwargs={
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "stream_options": {"include_usage": True},
            },
            timeout=300,
        ),
    )
    return p


def source_label(doc: Document) -> dict:
    return {
        "source_id": doc.meta.get("doc_id"),
        "title": doc.meta.get("title"),
        "page": doc.meta.get("page_number"),
        "url": doc.meta.get("url"),
    }


def build_messages(
    question: str, docs: list[Document], history: list[tuple[str, str]]
) -> list[ChatMessage]:
    """Prompt build: numbered sources in the system prompt, prior turns, then the question."""
    sources = "\n\n".join(
        f"[{i}] {d.meta.get('title')}, page {d.meta.get('page_number')}:\n{d.content.strip()}"
        for i, d in enumerate(docs, 1)
    )
    msgs = [ChatMessage.from_system(SYSTEM_PROMPT.format(sources=sources))]
    for role, text in history:
        msgs.append(
            ChatMessage.from_user(text)
            if role == "user"
            else ChatMessage.from_assistant(text)
        )
    # The 1.5B model drops citations unless the reminder sits next to the question.
    msgs.append(ChatMessage.from_user(question + CITE_REMINDER))
    return msgs


def cited(answer: str, docs: list[Document]) -> list[dict]:
    """Sources the answer actually cites, as [{n, title, page, url}]."""
    nums = sorted(
        {int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(docs)}
    )
    return [{"n": n, **source_label(docs[n - 1])} for n in nums]


class Rag:
    """Loads models once; the API calls retrieve() then generate()."""

    def __init__(self) -> None:
        store = document_store()
        self.retrieval = retrieval_pipeline(store)
        self.generation = generation_pipeline()
        self.retrieval.warm_up()

    def retrieve(self, question: str) -> list[Document]:
        out = self.retrieval.run(
            {
                "embed_query": {"text": question},
                "bm25_retrieve": {"query": question},
                "rerank": {"query": question},
            }
        )
        return out["rerank"]["documents"]

    def generate(
        self, messages: list[ChatMessage], on_token: Callable[[str], None]
    ) -> tuple[str, dict]:
        def cb(chunk: StreamingChunk) -> None:
            if chunk.content:
                on_token(chunk.content)

        reply = self.generation.run(
            {"generate": {"messages": messages, "streaming_callback": cb}}
        )["generate"]
        msg = reply["replies"][0]
        return msg.text or "", msg.meta.get("usage") or {}
