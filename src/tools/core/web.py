"""
Canonical Core Logic for Web Scraping and Search.
Resilient multi-engine search: DuckDuckGo HTML, Bing HTML scraper, DDGS, and Wikipedia fallback.
Includes query sanitization (quote stripping, operator normalization) and URL redirect unwrapping.
"""
from __future__ import annotations

import base64
import html
import logging
import re
import urllib.parse
from typing import Any

import httpx
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

from src.tools.core.contract import error_result, success_result

logger = logging.getLogger("hermes.tools.web")

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def unwrap_url(raw_url: str) -> str:
    """Unwrap redirect tracker URLs from search engines to get direct destination URLs."""
    if not raw_url:
        return ""
    clean = raw_url.strip()

    # DuckDuckGo redirect: //duckduckgo.com/l/?uddg=https%3A%2F%2F...
    if "duckduckgo.com/l/?" in clean and "uddg=" in clean:
        try:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(clean).query)
            dest = parsed.get("uddg", [""])[0]
            if dest and dest.startswith("http"):
                return urllib.parse.unquote(dest)
        except Exception:
            pass

    # Bing redirect: https://www.bing.com/ck/a?...&u=a1<base64>...
    if "bing.com/ck/a?" in clean and "u=a1" in clean:
        try:
            m = re.search(r"[?&]u=a1([^&]+)", clean)
            if m:
                raw_b64 = m.group(1)
                raw_b64 += "=" * (-len(raw_b64) % 4)
                decoded = base64.b64decode(raw_b64).decode("utf-8", errors="ignore")
                if decoded.startswith("http"):
                    return decoded
        except Exception:
            pass

    return clean


def sanitize_search_query(raw_query: str) -> str:
    """Clean and normalize a raw query string for web search engines.
    
    Removes wrapping/escaped quotes and prefixes that cause 0-result exact-match failures.
    """
    if not raw_query:
        return ""
    q = raw_query.strip()

    # Strip null bytes and control chars
    q = re.sub(r"[\x00-\x1f\x7f]", " ", q)

    # Strip prefixes like 'query: ' or 'search: '
    q = re.sub(r"^(?:query|search|q)\s*[:=]\s*", "", q, flags=re.IGNORECASE).strip()

    # Strip surrounding quotes
    q = re.sub(r"^[\"\'“”‘’]+|[\"\'“”‘’]+$", "", q).strip()

    # Normalize internal escaped quotes
    q = q.replace('\\"', '"').replace("\\'", "'")

    # If the query contains literal double or curly quotes, strip them to prevent exact-string misses
    if '"' in q or "“" in q or "”" in q:
        q = re.sub(r'[\"“”]', "", q)

    # Collapse multiple whitespaces
    q = re.sub(r"\s+", " ", q).strip()
    return q


def extract_fallback_keywords(query: str) -> str:
    """Extract essential keywords from conversational prompt sentences."""
    instructions = [
        r"\bin\s+\w+\s+bullet\s+points?\b.*",
        r"\band\s+cite\s+the\s+direct.*",
        r"\bstate\s+the\s+exact.*",
        r"\bsearch\s+(?:the\s+web\s+for|for|google\s+for)\b",
        r"\bplease\s+(?:find|tell\s+me|search|look\s+up)\b",
        r"\bcan\s+you\s+(?:find|search|tell\s+me)\b",
        r"\bgive\s+me\s+(?:the|all)?\b",
    ]
    cleaned = query
    for pat in instructions:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s\-\.]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else query


def _search_ddg_html(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Direct DuckDuckGo HTML endpoint query with standard browser headers."""
    headers = dict(_BROWSER_HEADERS)
    headers["Referer"] = "https://duckduckgo.com/"
    with httpx.Client(headers=headers, follow_redirects=True, timeout=5.0) as client:
        resp = client.post("https://html.duckduckgo.com/html/", data={"q": query})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[dict[str, str]] = []
        for body in soup.select(".result__body"):
            h2 = body.find("h2", class_="result__title")
            a_link = body.find("a", class_="result__a") or (h2.find("a") if h2 else None)
            if not a_link:
                continue
            title = a_link.get_text().strip()
            raw_href = a_link.get("href", "").strip()
            url = unwrap_url(raw_href)
            if not url.startswith("http"):
                continue
            snip_elem = body.find("a", class_="result__snippet")
            snippet = snip_elem.get_text().strip() if snip_elem else ""
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })
            if len(results) >= limit:
                break
        return results


def _search_bing(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Bing web search scraper with direct URL unwrap."""
    with httpx.Client(headers=_BROWSER_HEADERS, follow_redirects=True, timeout=5.0) as client:
        resp = client.get("https://www.bing.com/search", params={"q": query})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[dict[str, str]] = []
        for li in soup.select("li.b_algo"):
            h2 = li.find("h2")
            if not h2:
                continue
            a_link = h2.find("a")
            if not a_link or not a_link.get("href"):
                continue
            title = a_link.get_text().strip()
            raw_href = a_link["href"].strip()
            url = unwrap_url(raw_href)
            if not url.startswith("http"):
                continue
            p = li.find("p")
            snippet = p.get_text().strip() if p else ""
            if not snippet:
                cap = li.find("div", class_="b_caption")
                if cap:
                    snippet = cap.get_text().strip()
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })
            if len(results) >= limit:
                break
        return results


def _search_wikipedia(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Wikipedia Search API for factual/entity queries."""
    headers = {"User-Agent": "HermesAgent/1.0 (info@hermes.ai)"}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=4.0) as client:
        resp = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": limit,
            },
        )
        if resp.status_code != 200:
            return []
        try:
            data = resp.json().get("query", {}).get("search", [])
        except Exception:
            return []
        results: list[dict[str, str]] = []
        for item in data:
            title = item.get("title", "")
            clean_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            clean_snippet = html.unescape(clean_snippet).strip()
            slug = title.replace(" ", "_")
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(slug)
            results.append({
                "title": title,
                "url": url,
                "snippet": clean_snippet,
            })
            if len(results) >= limit:
                break
        return results


def _search_ddgs(query: str, limit: int = 5, max_retries: int = 2) -> list[dict[str, str]]:
    """Fallback to DDGS Python library if installed."""
    if DDGS is None:
        return []
    for attempt in range(max(1, max_retries)):
        try:
            with DDGS(timeout=5) as ddgs:
                raw = list(ddgs.text(query, max_results=limit))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": unwrap_url(r.get("href", "")),
                        "snippet": r.get("body", ""),
                    }
                    for r in raw
                    if r.get("href")
                ]
        except Exception as exc:
            logger.debug("DDGS attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                return []
    return []


def search(query: str, max_results: int = 5, max_retries: int = 2) -> dict[str, Any]:
    """Resilient web search across multiple search backends with automatic sanitization."""
    tool_name = "web_scraper.search"
    if not query or not query.strip():
        return error_result("INVALID_ARGUMENT", "Search query cannot be empty", tool_name)

    limit = min(max(1, max_results), 10)
    cleaned_query = sanitize_search_query(query)
    if not cleaned_query:
        return error_result("INVALID_ARGUMENT", "Search query cannot be empty", tool_name)

    results: list[dict[str, str]] = []

    # 1. Primary: DuckDuckGo HTML
    try:
        ddg_res = _search_ddg_html(cleaned_query, limit=limit)
        results.extend(ddg_res)
    except Exception as exc:
        logger.debug("DDG HTML search failed: %s", exc)

    # 2. Secondary: Bing Web Search (fallback or supplement up to limit)
    if len(results) < limit:
        try:
            bing_res = _search_bing(cleaned_query, limit=limit - len(results))
            existing_urls = {r.get("url", "").rstrip("/") for r in results}
            for br in bing_res:
                if br.get("url", "").rstrip("/") not in existing_urls:
                    results.append(br)
                    existing_urls.add(br.get("url", "").rstrip("/"))
        except Exception as exc:
            logger.debug("Bing search failed: %s", exc)

    # 3. Tertiary: DDGS library
    if not results:
        try:
            results = _search_ddgs(cleaned_query, limit=limit, max_retries=max_retries)
        except Exception as exc:
            logger.debug("DDGS search failed: %s", exc)

    # 4. Quaternary: Keyword simplification fallback (retry DDG HTML & Bing)
    if not results:
        fallback_query = extract_fallback_keywords(cleaned_query)
        if fallback_query != cleaned_query:
            try:
                results = _search_ddg_html(fallback_query, limit=limit)
            except Exception:
                pass
            if not results:
                try:
                    results = _search_bing(fallback_query, limit=limit)
                except Exception:
                    pass

    # 5. Quinary: Wikipedia Search API for entity/factual terms
    if not results:
        try:
            results = _search_wikipedia(cleaned_query, limit=limit)
        except Exception as exc:
            logger.debug("Wikipedia search failed: %s", exc)

    # Format output items with both url/href and snippet/body for maximum client compatibility
    formatted = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "href": r.get("url", ""),
            "snippet": r.get("snippet", ""),
            "body": r.get("snippet", ""),
        }
        for r in results[:limit]
    ]

    msg = "Search completed successfully." if formatted else "No verified search results found."
    return success_result(formatted, tool_name, query=query, count=len(formatted), message=msg)


def read_url(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """Fetch and extract text content from a web URL."""
    tool_name = "web_scraper.read_url"
    clean_url = unwrap_url(url)
    if not clean_url or not clean_url.strip().startswith("http"):
        return error_result("INVALID_URL", f"Invalid URL scheme: '{url}'", tool_name)

    clean_url = clean_url.strip()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_BROWSER_HEADERS) as client:
            resp = client.get(clean_url)
            if resp.status_code == 404:
                return error_result("NOT_FOUND", f"Webpage returned 404 Not Found: {clean_url}", tool_name)
            if resp.status_code == 403:
                return error_result("FORBIDDEN", f"Webpage returned 403 Forbidden: {clean_url}", tool_name, retryable=True)
            if resp.status_code >= 400:
                return error_result("HTTP_ERROR", f"HTTP {resp.status_code} while fetching {clean_url}", tool_name)

            content_type = resp.headers.get("content-type", "").lower()
            is_pdf = "application/pdf" in content_type or clean_url.lower().endswith(".pdf")
            if is_pdf:
                pdf_raw = resp.content.decode("latin-1", errors="ignore")
                words = re.findall(r"[A-Za-z0-9\.\,\:\;\%\$\-\+\(\)\s]{4,}", pdf_raw)
                cleaned = " ".join(" ".join(words).split()).strip()
                cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
                content = cleaned[:3500] if cleaned else "[PDF document returned empty text content]"
                return success_result(content, tool_name, url=clean_url, status_code=resp.status_code)

            html_text = resp.text
            soup = BeautifulSoup(html_text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
                tag.extract()
            text = soup.get_text(separator=" ")
            cleaned = " ".join(text.split()).strip()
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
            content = cleaned[:3500] if cleaned else "[Webpage returned empty text content]"
            return success_result(content, tool_name, url=clean_url, status_code=resp.status_code)
    except httpx.TimeoutException:
        return error_result("TIMEOUT", f"Request timed out while fetching {clean_url}", tool_name, retryable=True)
    except Exception as exc:
        return error_result("EXECUTION_ERROR", f"Failed to read webpage {clean_url}: {exc}", tool_name)
