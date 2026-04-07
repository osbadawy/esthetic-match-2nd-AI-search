#!/usr/bin/env python3
"""
WebRetriever: keyless web search + fetch for HF CPU RAG.

Improvements:
- Decodes DuckDuckGo redirect URLs (/l/?uddg=...)
- Extracts paragraph/list focused text (less noisy than full-page)
- Supports max_chars_per_doc
- Gentle delay + graceful failures
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup


@dataclass
class WebDoc:
    title: str
    url: str
    snippet: str


class WebRetriever:
    def __init__(
        self,
        user_agent: str = None,
        timeout_sec: int = 15,
        polite_delay_sec: float = 0.35,
    ):
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; AestheticRAG/1.0)"
        self.timeout_sec = int(timeout_sec)
        self.polite_delay_sec = float(polite_delay_sec)

    def _decode_ddg_url(self, href: str) -> str:
        if not href:
            return ""
        try:
            p = urlparse(href)
            if "duckduckgo.com" in (p.netloc or "") and p.path.startswith("/l/"):
                qs = parse_qs(p.query or "")
                if "uddg" in qs and qs["uddg"]:
                    return unquote(qs["uddg"][0])
        except Exception:
            pass
        return href

    def search(self, query: str, max_results: int = 5) -> List[WebDoc]:
        q = (query or "").strip()
        if not q:
            return []

        url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"
        headers = {"User-Agent": self.user_agent}

        r = requests.get(url, headers=headers, timeout=self.timeout_sec)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        results: List[WebDoc] = []

        for a in soup.select("a.result__a")[: max_results * 3]:
            title = a.get_text(" ", strip=True)
            href = self._decode_ddg_url(a.get("href") or "")
            if not title or not href:
                continue
            results.append(WebDoc(title=title, url=href, snippet=""))
            if len(results) >= max_results:
                break

        time.sleep(self.polite_delay_sec)
        return results

    def fetch_snippet(self, url: str, max_chars: int = 1200) -> str:
        headers = {"User-Agent": self.user_agent}
        r = requests.get(url, headers=headers, timeout=self.timeout_sec)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove obvious noise
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # Prefer paragraph/list items (higher info density)
        chunks = []
        for el in soup.find_all(["p", "li"]):
            t = el.get_text(" ", strip=True)
            if t and len(t) >= 40:
                chunks.append(t)

        text = " ".join(chunks) if chunks else soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "…"

        time.sleep(self.polite_delay_sec)
        return text

    def search_and_fetch(
        self,
        queries: List[str],
        max_results_per_query: int = 3,
        max_docs: int = 6,
        max_chars_per_doc: int = 1200,
    ) -> List[WebDoc]:
        docs: List[WebDoc] = []
        seen = set()

        for q in queries:
            q = (q or "").strip()
            if not q:
                continue

            try:
                results = self.search(q, max_results=max_results_per_query)
            except Exception:
                results = []

            for res in results:
                try:
                    p = urlparse(res.url)
                    key = (p.netloc.lower(), p.path.lower())
                except Exception:
                    key = res.url

                if key in seen:
                    continue
                seen.add(key)

                try:
                    snippet = self.fetch_snippet(res.url, max_chars=int(max_chars_per_doc))
                except Exception:
                    snippet = ""

                docs.append(WebDoc(title=res.title, url=res.url, snippet=snippet))
                if len(docs) >= max_docs:
                    return docs

        return docs
