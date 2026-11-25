"""
Unit tests for resilient web search and scraper capabilities.
Verifies query sanitization, URL unwrapping, multi-engine fallbacks, and output formatting.
"""
import pytest
from src.tools.core.web import (
    extract_fallback_keywords,
    sanitize_search_query,
    search,
    unwrap_url,
)
from src.tools.validators.web_validator import WebSearchValidator, ValidationStatus


def test_sanitize_search_query_quotes():
    """Verify enclosing and internal quotes are properly stripped to prevent 0-match search bugs."""
    assert sanitize_search_query('"Aakasam lo Oka Thara" box office') == "Aakasam lo Oka Thara box office"
    assert sanitize_search_query("'how many ai startups in SF'") == "how many ai startups in SF"
    assert sanitize_search_query('“machine learning papers”') == "machine learning papers"
    assert sanitize_search_query('query: "latest tech news"') == "latest tech news"
    assert sanitize_search_query('q:   test query   ') == "test query"


def test_unwrap_url_bing_and_ddg():
    """Verify redirect wrapper URLs from search engines are cleanly decoded."""
    # Bing base64 redirect: u=a1<base64>
    # "https://www.sacnilk.com/test" in base64 is "aHR0cHM6Ly93d3cuc2Fjbmlsay5jb20vdGVzdA"
    bing_url = "https://www.bing.com/ck/a?!&&p=123&u=a1aHR0cHM6Ly93d3cuc2Fjbmlsay5jb20vdGVzdA&ntb=1"
    assert unwrap_url(bing_url) == "https://www.sacnilk.com/test"

    # DDG redirect
    ddg_url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fid%3D1&rut=xyz"
    assert unwrap_url(ddg_url) == "https://example.com/page?id=1"

    # Direct URL
    direct = "https://ycombinator.com/companies"
    assert unwrap_url(direct) == direct


def test_extract_fallback_keywords():
    """Verify prompt boilerplate is cleanly stripped for keyword fallback queries."""
    prompt = "Search the web for the Q3 2024 earnings releases for both Spotify and Netflix. In two bullet points, state the exact year-over-year revenue growth percentage"
    keywords = extract_fallback_keywords(prompt)
    assert "bullet points" not in keywords.lower()
    assert "search the web" not in keywords.lower()
    assert "spotify" in keywords.lower()
    assert "netflix" in keywords.lower()


def test_search_empty_query():
    """Verify empty search query returns error result."""
    res = search("")
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_ARGUMENT"


def test_search_live_query_with_quotes():
    """Verify searching with quoted movie title yields verified results and passes WebSearchValidator."""
    res = search('"Aakasam lo Oka Thara" box office', max_results=3)
    assert res["success"] is True
    assert len(res["data"]) > 0

    first = res["data"][0]
    assert "title" in first
    assert "url" in first
    assert "snippet" in first
    assert first["url"].startswith("http")

    # Validate output with WebSearchValidator
    validator = WebSearchValidator()
    val_res = validator.validate(
        "web_scraper.search",
        {"query": '"Aakasam lo Oka Thara" box office'},
        res,
    )
    assert val_res.status == ValidationStatus.VERIFIED
    assert val_res.is_valid is True
    assert val_res.confidence >= 0.7


def test_search_sf_startups_query():
    """Verify query 'how many ai startups are in San Francisco' returns verified results."""
    res = search("how many ai startups are in San Francisco", max_results=3)
    assert res["success"] is True
    assert len(res["data"]) > 0

    validator = WebSearchValidator()
    val_res = validator.validate(
        "web_scraper.search",
        {"query": "how many ai startups are in San Francisco"},
        res,
    )
    assert val_res.status == ValidationStatus.VERIFIED
    assert val_res.is_valid is True
