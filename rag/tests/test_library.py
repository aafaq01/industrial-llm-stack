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


def _docx(paragraphs, tab=False):
    import io
    import zipfile

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    from xml.sax.saxutils import escape

    body = "".join(
        f"<w:p><w:r><w:t>{escape(t)}</w:t></w:r>{'<w:r><w:tab/></w:r>' if tab else ''}</w:p>"
        for t in paragraphs
    )
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_word_files_are_read():
    title, text, is_pdf = library.extract(
        "plan.docx", _docx(["First line", "Second & third"])
    )
    assert title == "plan.docx" and not is_pdf
    assert text == "First line\nSecond & third"


def test_other_zip_files_are_still_refused(monkeypatch):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", "<x/>")
    with pytest.raises(library.Rejected, match="not supported"):
        library.extract("sheet.xlsx", buf.getvalue())
    with pytest.raises(library.Rejected, match="not supported"):
        library.extract("broken.docx", b"PK\x03\x04 not really a zip")


def test_word_zip_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(library, "MAX_BYTES", 100)
    with pytest.raises(library.Rejected, match="too large"):
        library.extract("big.docx", _docx(["x" * 5000]))


def test_damaged_word_file_gets_a_clear_message():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><unclosed>")
    with pytest.raises(library.Rejected, match="damaged"):
        library.extract("bad.docx", buf.getvalue())


def test_word_table_rows_stay_together():
    import io
    import zipfile

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def cell(t):
        return f"<w:tc><w:p><w:r><w:t>{t}</w:t></w:r></w:p></w:tc>"

    xml = (
        f'<w:document xmlns:w="{w}"><w:body><w:p><w:r><w:t>Before</w:t></w:r></w:p>'
        f"<w:tbl><w:tr>{cell('Part')}{cell('Number')}</w:tr><w:tr>{cell('Seal kit')}{cell('ZK-4471')}</w:tr></w:tbl>"
        "<w:p><w:r><w:t>After</w:t></w:r></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    _, text, _ = library.extract("t.docx", buf.getvalue())
    assert text.splitlines() == [
        "Before",
        "Part | Number",
        "Seal kit | ZK-4471",
        "After",
    ]
