"""Own-documents helpers that need no Elasticsearch or models."""

import pytest
from app import library


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9200/",
        "http://localhost/",
        "http://10.1.2.3/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:10.0.0.1]/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://",
    ],
)
def test_private_and_odd_urls_are_rejected(url):
    with pytest.raises(library.Rejected):
        library.check_url(url)


def test_private_urls_can_be_allowed(monkeypatch):
    monkeypatch.setattr(library, "ALLOW_PRIVATE_URLS", True)
    library.check_url("http://127.0.0.1:9200/")  # no exception


def test_html_keeps_text_and_title_and_drops_scripts():
    html = b"<html><head><title> My  page </title><style>p{}</style></head><body><script>evil()</script><p>Hello</p><p>World</p></body></html>"
    title, text, is_pdf = library.extract("x.html", html)
    assert title == "My page" and not is_pdf
    assert (
        "Hello" in text and "World" in text and "evil" not in text and "p{}" not in text
    )


def test_plain_text_and_markdown_pass_through():
    title, text, is_pdf = library.extract(
        "notes.md", "# Heading\n\nSome é text".encode()
    )
    assert title == "notes.md" and "Some é text" in text and not is_pdf


@pytest.mark.parametrize("data", [b"PK\x03\x04docx", b"\x89PNG\r\n\x1a\n\x00\x00"])
def test_binary_files_are_rejected(data):
    with pytest.raises(library.Rejected, match="not supported"):
        library.extract("file.bin", data)


def test_empty_text_is_rejected_before_any_model_loads():
    with pytest.raises(library.Rejected, match="No text"):
        library.index_text("id", "t", None, " \f \n", True)
