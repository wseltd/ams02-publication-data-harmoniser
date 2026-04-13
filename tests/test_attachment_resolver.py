"""Tests for ams02wb.crawler.attachment_resolver."""

from ams02wb.crawler.attachment_resolver import resolve_attachments

BASE_URL = "https://ams02.space/publications/paper42/"


def test_resolve_csv_attachment_absolute_url() -> None:
    html = '<html><body><a href="https://ams02.space/data/flux.csv">CSV</a></body></html>'
    result = resolve_attachments(html, BASE_URL)
    assert len(result) == 1
    assert result[0].url == "https://ams02.space/data/flux.csv"
    assert result[0].filename == "flux.csv"
    assert result[0].content_type == "csv"


def test_resolve_pdf_attachment_absolute_url() -> None:
    html = '<html><body><a href="https://ams02.space/papers/paper.pdf">PDF</a></body></html>'
    result = resolve_attachments(html, BASE_URL)
    assert len(result) == 1
    assert result[0].url == "https://ams02.space/papers/paper.pdf"
    assert result[0].filename == "paper.pdf"
    assert result[0].content_type == "pdf"


def test_resolve_relative_href_against_base_url() -> None:
    html = '<html><body><a href="../data/table.csv">table</a></body></html>'
    result = resolve_attachments(html, BASE_URL)
    assert len(result) == 1
    assert result[0].url == "https://ams02.space/publications/data/table.csv"
    assert result[0].filename == "table.csv"
    assert result[0].content_type == "csv"


def test_resolve_strips_query_params_from_filename() -> None:
    html = '<html><body><a href="https://ams02.space/dl/results.csv?v=2&token=abc">DL</a></body></html>'
    result = resolve_attachments(html, BASE_URL)
    assert len(result) == 1
    assert result[0].filename == "results.csv"
    assert "?" not in result[0].filename


def test_resolve_case_insensitive_extension_matching() -> None:
    html = (
        "<html><body>"
        '<a href="https://ams02.space/A.CSV">a</a>'
        '<a href="https://ams02.space/B.Pdf">b</a>'
        "</body></html>"
    )
    result = resolve_attachments(html, BASE_URL)
    assert len(result) == 2
    assert result[0].content_type == "csv"
    assert result[1].content_type == "pdf"


def test_resolve_returns_empty_list_for_no_attachments() -> None:
    html = '<html><body><a href="https://example.com/page.html">link</a><p>No files</p></body></html>'
    result = resolve_attachments(html, BASE_URL)
    assert result == []


def test_resolve_mixed_csv_and_pdf_links() -> None:
    html = (
        "<html><body>"
        '<a href="flux.csv">CSV</a>'
        '<a href="https://other.org/readme.txt">TXT</a>'
        '<a href="supplement.pdf">PDF</a>'
        "</body></html>"
    )
    result = resolve_attachments(html, BASE_URL)
    assert len(result) == 2
    assert result[0].content_type == "csv"
    assert result[0].filename == "flux.csv"
    assert result[1].content_type == "pdf"
    assert result[1].filename == "supplement.pdf"
