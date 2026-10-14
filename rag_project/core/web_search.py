"""External search adapter with strict URL validation."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlparse

from .errors import WebSearchError


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str
    domain: str


def _validated_result(raw: dict) -> SearchResult | None:
    href = raw.get("href") or raw.get("url")
    if not isinstance(href, str):
        return None
    if len(href) > 2000 or any(ord(char) < 32 or char.isspace() for char in href):
        return None
    try:
        parsed = urlparse(href)
        host = parsed.hostname
        parsed.port
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    if parsed.username or parsed.password:
        return None
    host = host.lower().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
        if not address.is_global:
            return None
    except ValueError:
        if host == "localhost" or "." not in host:
            return None
    title = str(raw.get("title") or "").strip()
    snippet = str(raw.get("body") or raw.get("snippet") or "").strip()
    if not title:
        return None
    return SearchResult(title[:300], snippet[:1200], href, host)


def search_web(query: str, count: int = 5) -> list[SearchResult]:
    if not query.strip() or len(query) > 300:
        raise ValueError("Enter a search query between 1 and 300 characters.")
    if not 1 <= count <= 10:
        raise ValueError("Choose between 1 and 10 search results.")
    try:
        from ddgs import DDGS

        raw_results = DDGS(timeout=10).text(query, max_results=count)
        if raw_results is None:
            return []
        results = []
        for raw in raw_results:
            if isinstance(raw, dict):
                result = _validated_result(raw)
                if result:
                    results.append(result)
        return results
    except ImportError as exc:
        raise WebSearchError(
            "The web search package is missing. Install the project requirements and retry."
        ) from exc
    except Exception as exc:
        raise WebSearchError(
            "Web search is temporarily unavailable. Check your internet connection and try again."
        ) from exc
