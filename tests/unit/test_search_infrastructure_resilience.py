"""
Tests for Web Search and Content Extraction Resilience in Hermes Agent Platform.

Covers:
1. Normal search query execution and contract verification.
2. Query containing conversational wrappers sanitization.
3. Search returning zero results (safe handling and explicit message).
4. Search backend unavailable (graceful fallback across engines).
5. Multiple search engines returning different results (merging and deduplication).
6. Redirect URL decoding (Bing base64 and DuckDuckGo uddg parameter unwrapping).
7. PDF document text extraction and control code cleaning.
8. Malformed webpage content (broken HTML, script extraction, human text parsing).
9. Invalid and null-byte content handling (sanitization preventing downstream errors).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.tools.core.web import (
    _search_ddg_html,
    _search_bing,
    extract_fallback_keywords,
    read_url,
    sanitize_search_query,
    search,
    unwrap_url,
)
from src.tools.validators.web_validator import WebSearchValidator, ValidationStatus


# ---------------------------------------------------------------------------
# 1. Normal search query
# ---------------------------------------------------------------------------
def test_search_normal_query():
    """Verify standard search query returns formatted results with title, URL, and snippet."""
    mock_results = [
        {"title": "Quantum Error Correction Guide", "url": "https://example.com/quantum", "snippet": "A deep dive into surface codes and logical qubits."},
        {"title": "Surface Codes Overview", "url": "https://example.com/surface-codes", "snippet": "Fault-tolerant quantum memory architecture."},
    ]

    with patch("src.tools.core.web._search_ddg_html", return_value=mock_results):
        res = search("quantum error correction surface codes", max_results=2)
        assert res["success"] is True
        assert res["metadata"]["tool_name"] == "web_scraper.search"
        assert res["metadata"]["count"] == 2
        assert len(res["data"]) == 2

        first = res["data"][0]
        assert first["title"] == "Quantum Error Correction Guide"
        assert first["url"] == "https://example.com/quantum"
        assert first["href"] == "https://example.com/quantum"
        assert "surface codes" in first["snippet"]


# ---------------------------------------------------------------------------
# 2. Query containing conversational wrappers
# ---------------------------------------------------------------------------
def test_search_query_containing_conversational_wrappers():
    """Verify prompt boilerplate and conversational wrappers are stripped before engine dispatch."""
    wrapper_query = (
        "Please search the web for the Q3 2024 earnings of Apple. "
        "In two bullet points, state the exact year-over-year revenue and cite the direct source."
    )
    cleaned = sanitize_search_query(wrapper_query)
    keywords = extract_fallback_keywords(cleaned)

    # Conversational phrases are removed
    assert "in two bullet points" not in keywords.lower()
    assert "cite the direct" not in keywords.lower()
    assert "please search" not in keywords.lower()
    # Core entity and subject are preserved
    assert "apple" in keywords.lower()
    assert "earnings" in keywords.lower()

    # Also test prefix stripping like "query: "
    assert sanitize_search_query('query: "latest AI news"') == "latest AI news"
    assert sanitize_search_query('search: machine learning') == "machine learning"


# ---------------------------------------------------------------------------
# 3. Search returning zero results
# ---------------------------------------------------------------------------
def test_search_returning_zero_results():
    """Verify zero search results are safely handled without crashes, returning explicit count=0."""
    with patch("src.tools.core.web._search_ddg_html", return_value=[]), \
         patch("src.tools.core.web._search_bing", return_value=[]), \
         patch("src.tools.core.web._search_ddgs", return_value=[]), \
         patch("src.tools.core.web._search_wikipedia", return_value=[]):

        res = search("xyznonexistentterm123456789abc")
        assert res["success"] is True
        assert res["data"] == []
        assert res["metadata"]["count"] == 0
        assert "No verified search results found." in res["metadata"]["message"]


# ---------------------------------------------------------------------------
# 4. Search backend unavailable
# ---------------------------------------------------------------------------
def test_search_backend_unavailable_fallback():
    """Verify that when the primary search backend throws a connection error, it falls back to secondary."""
    bing_results = [
        {"title": "Bing Result", "url": "https://bing-source.com/page", "snippet": "Recovered via secondary search backend."},
    ]

    def mock_ddg_fail(*args, **kwargs):
        raise httpx.ConnectError("Failed to establish connection to html.duckduckgo.com")

    with patch("src.tools.core.web._search_ddg_html", side_effect=mock_ddg_fail), \
         patch("src.tools.core.web._search_bing", return_value=bing_results):

        res = search("resilience test", max_results=3)
        assert res["success"] is True
        assert len(res["data"]) == 1
        assert res["data"][0]["title"] == "Bing Result"
        assert res["data"][0]["url"] == "https://bing-source.com/page"

    # If all backends throw exceptions, returns clean 0-result response without crashing
    with patch("src.tools.core.web._search_ddg_html", side_effect=httpx.ConnectError("DDG down")), \
         patch("src.tools.core.web._search_bing", side_effect=httpx.ConnectTimeout("Bing timeout")), \
         patch("src.tools.core.web._search_ddgs", side_effect=RuntimeError("DDGS failure")), \
         patch("src.tools.core.web._search_wikipedia", side_effect=Exception("Wiki error")):

        res_all_down = search("critical search term")
        assert res_all_down["success"] is True
        assert res_all_down["data"] == []
        assert res_all_down["metadata"]["count"] == 0


# ---------------------------------------------------------------------------
# 5. Multiple search engines returning different results
# ---------------------------------------------------------------------------
def test_multiple_search_engines_returning_different_results():
    """Verify results from multiple engines are unified, supplemented up to limit, and deduplicated."""
    ddg_res = [
        {"title": "DDG Item 1", "url": "https://example.com/article-1", "snippet": "Snippet 1"},
        {"title": "DDG Duplicate", "url": "https://example.com/shared-url", "snippet": "DDG Snippet"},
    ]
    bing_res = [
        {"title": "Bing Duplicate", "url": "https://example.com/shared-url", "snippet": "Bing Snippet"},
        {"title": "Bing Item 2", "url": "https://example.com/article-2", "snippet": "Snippet 2"},
    ]

    with patch("src.tools.core.web._search_ddg_html", return_value=ddg_res), \
         patch("src.tools.core.web._search_bing", return_value=bing_res):

        res = search("multi-engine query", max_results=5)
        assert res["success"] is True
        urls = [r["url"] for r in res["data"]]
        # URLs must be deduplicated
        assert len(urls) == len(set(urls))
        assert "https://example.com/article-1" in urls
        assert "https://example.com/shared-url" in urls
        assert "https://example.com/article-2" in urls


# ---------------------------------------------------------------------------
# 6. Redirect URL decoding
# ---------------------------------------------------------------------------
def test_redirect_url_decoding():
    """Verify search engine tracking and redirect wrappers are unwrapped into direct canonical URLs."""
    # Bing base64 redirect: u=a1<base64>
    # "https://www.nature.com/articles/s41586" in base64: "aHR0cHM6Ly93d3cubmF0dXJlLmNvbS9hcnRpY2xlcy9zNDE1ODY"
    bing_redirect = "https://www.bing.com/ck/a?!&&p=abc&u=a1aHR0cHM6Ly93d3cubmF0dXJlLmNvbS9hcnRpY2xlcy9zNDE1ODY&ntb=1"
    assert unwrap_url(bing_redirect) == "https://www.nature.com/articles/s41586"

    # DuckDuckGo redirect: uddg parameter
    ddg_redirect = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Farxiv.org%2Fabs%2F2401.00001&rut=test"
    assert unwrap_url(ddg_redirect) == "https://arxiv.org/abs/2401.00001"

    # Direct URL passes through unchanged
    direct_url = "https://github.com/google-deepmind"
    assert unwrap_url(direct_url) == direct_url


# ---------------------------------------------------------------------------
# 7. PDF extraction
# ---------------------------------------------------------------------------
def test_pdf_extraction():
    """Verify read_url extracts text from PDF responses and strips non-printable stream codes."""
    pdf_payload = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"stream\n"
        b"Empirical Evaluation of Large Language Model Reasoning and Mathematical Proofs.\n"
        b"Abstract: This research paper analyzes multi-step reasoning capabilities in transformers.\n"
        b"endstream\n"
        b"%%EOF"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/pdf"}
    mock_resp.content = pdf_payload

    with patch("httpx.Client.get", return_value=mock_resp):
        res = read_url("https://arxiv.org/pdf/2401.00001.pdf")
        assert res["success"] is True
        assert res["metadata"]["tool_name"] == "web_scraper.read_url"
        content = res["data"]
        assert "Empirical Evaluation of Large Language Model" in content
        assert "Abstract" in content


# ---------------------------------------------------------------------------
# 8. Malformed webpage content
# ---------------------------------------------------------------------------
def test_malformed_webpage_content():
    """Verify malformed HTML, nested broken tags, scripts, and navigation clutter are sanitized."""
    malformed_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Malformed Page</title>
        <script type="text/javascript">
          function maliciousCode() { alert("malicious script should be stripped"); }
          var tracker = { id: 12345 };
        </script>
        <style>
          body { background: red; }
          .hidden { display: none; }
        </style>
      </head>
      <body>
        <nav><a href="/home">Nav link</a></nav>
        <header><h1>Header Navigation</h1></header>
        <div>
          <p>This is the actual article content regarding semiconductor lithography advancements.
          <div><span>Nested unclosed span and paragraph text.
          <p>Additional paragraph without proper closing tags.
        </div>
        <footer>Copyright 2026 Example Corp</footer>
      </body>
    </html>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
    mock_resp.text = malformed_html

    with patch("httpx.Client.get", return_value=mock_resp):
        res = read_url("https://example.com/malformed-article")
        assert res["success"] is True
        content = res["data"]
        # Actual article text is present
        assert "semiconductor lithography advancements" in content
        assert "Nested unclosed span" in content
        # Scripts, styles, and header/footer elements are stripped
        assert "function maliciousCode" not in content
        assert "background: red" not in content
        assert "Copyright 2026" not in content


# ---------------------------------------------------------------------------
# 9. Invalid/null-byte content
# ---------------------------------------------------------------------------
def test_invalid_null_byte_content_sanitization():
    """Verify null bytes (\x00) and ASCII control characters are stripped from webpage content."""
    corrupted_html = (
        "<html><body>"
        "Valid text prefix. \x00\x01\x02Corrupted text with embedded null bytes\x00 and bell codes\x07. "
        "Valid text suffix."
        "</body></html>"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.text = corrupted_html

    with patch("httpx.Client.get", return_value=mock_resp):
        res = read_url("https://example.com/null-byte-page")
        assert res["success"] is True
        content = res["data"]
        # Null bytes and control characters must be purged
        assert "\x00" not in content
        assert "\x07" not in content
        assert "\x01" not in content
        assert "Valid text prefix" in content
        assert "Valid text suffix" in content
