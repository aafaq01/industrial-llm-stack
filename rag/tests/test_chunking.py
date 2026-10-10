from app.rag import chunker
from haystack import Document


def run(text: str) -> list[Document]:
    return chunker().run(
        {"clean": {"documents": [Document(content=text, meta={"title": "T"})]}}
    )["split"]["documents"]


def test_chunks_are_bounded_overlapping_and_keep_page_numbers():
    page1 = " ".join(f"a{i}" for i in range(200))
    page2 = " ".join(f"b{i}" for i in range(200))
    chunks = run(f"{page1}\f{page2}")

    # the splitter counts space-separated units, so "a199\fb0" is one word
    assert all(len(c.content.strip().split(" ")) <= 150 for c in chunks)
    assert chunks[0].meta["page_number"] == 1 and chunks[-1].meta["page_number"] == 2
    assert chunks[0].meta["title"] == "T"
    # 30-word overlap: the last 30 words of chunk 0 start chunk 1
    assert (
        chunks[0].content.strip().split(" ")[-30:]
        == chunks[1].content.strip().split(" ")[:30]
    )
    words = {w for c in chunks for w in c.content.split()}
    assert {"a0", "a199", "b0", "b199"} <= words


def test_cleaning_drops_nist_footer_and_blank_lines():
    text = "Intro line\n\n\nThis publication is available free of charge from: https://doi.org/10.6028/NIST.TN.1900\nBody"
    [chunk] = run(text)
    assert "available free of charge" not in chunk.content
    assert "\n\n" not in chunk.content
    assert "Intro line" in chunk.content and "Body" in chunk.content
