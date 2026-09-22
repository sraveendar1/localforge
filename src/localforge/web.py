"""Web access for the orchestrator: `web_search` and `fetch_url`.

Local models get no internet access of their own -- Ollama is text in, text
out. Instead the frontier model looks things up through these two tools and
passes what it found into each delegated subtask's instructions. They run
inside localforge itself, so they behave the same whether the orchestrator is
reached through an API key (a plain completion call with no built-in tools)
or through a provider's CLI (whose own built-in tools are switched off; see
cli_transport).

No API key needed: search uses DuckDuckGo's plain-HTML endpoint. That
endpoint occasionally rate-limits; the tool then says so instead of failing,
and the orchestrator can fall back to fetch_url on a URL it already knows.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

SEARCH_URL = "https://html.duckduckgo.com/html/"
MAX_RESULTS = 6
MAX_PAGE_CHARS = 12_000  # per fetch: enough to be useful, small enough for context
MAX_DOWNLOAD_BYTES = 3_000_000
MAX_REDIRECTS = 5
TIMEOUT = 20.0
USER_AGENT = "Mozilla/5.0 (compatible; localforge/0.1; +https://github.com/sraveendar1/localforge)"


class WebError(RuntimeError):
    """A fetch/search failed in a way worth telling the orchestrator about."""


# --- HTML -> readable text ---------------------------------------------------

_SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe"}
_BLOCK = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "hr", "main", "nav", "dd", "dt",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in _SKIP:
            self._skip_depth += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """(title, readable text) from an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    text = "\n".join(line for line in lines if line)
    return " ".join(parser.title.split()), text


# --- fetching, with a guard against reaching the local network ---------------


def _check_public(url: str) -> None:
    """Refuse anything but http(s) to a public address. Page text flows back
    into the orchestrator, so a page (or a prompt) could otherwise steer it
    into probing this machine or the local network -- a router admin page,
    Ollama's own API on localhost, cloud metadata endpoints.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise WebError(f"only http(s) URLs can be fetched, not {url!r}")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise WebError(f"could not resolve {parsed.hostname}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise WebError(f"refusing to fetch {parsed.hostname}: it resolves to a non-public address ({ip})")


@dataclass
class _Page:
    url: str
    status: int
    content_type: str
    text: str


def _get(url: str) -> _Page:
    """GET with redirects followed by hand, so every hop is checked, and the
    body capped at MAX_DOWNLOAD_BYTES.
    """
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _check_public(url)
            with client.stream("GET", url) as resp:
                if resp.is_redirect and "location" in resp.headers:
                    url = urljoin(url, resp.headers["location"])
                    continue
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_DOWNLOAD_BYTES:
                        break
                return _Page(
                    url=str(resp.url),
                    status=resp.status_code,
                    content_type=resp.headers.get("content-type", "").lower(),
                    text=bytes(body).decode(resp.encoding or "utf-8", errors="replace"),
                )
    raise WebError(f"too many redirects fetching {url}")


def fetch_url(url: str) -> str:
    """Readable text of one web page (or plain-text/JSON document)."""
    url = url.strip()
    try:
        page = _get(url)
    except httpx.HTTPError as exc:
        raise WebError(f"could not fetch {url}: {exc}") from exc
    if page.status >= 400:
        raise WebError(f"{url} returned HTTP {page.status}")

    ctype = page.content_type
    if "html" in ctype or (not ctype and page.text.lstrip().startswith("<")):
        title, text = html_to_text(page.text)
    elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
        title, text = "", page.text
    else:
        raise WebError(f"{url} is {ctype or 'an unknown type'}, not a text page")

    truncated = len(text) > MAX_PAGE_CHARS
    text = text[:MAX_PAGE_CHARS]
    header = f"URL: {page.url}\n" + (f"Title: {title}\n" if title else "")
    note = f"\n\n[truncated to the first {MAX_PAGE_CHARS} characters]" if truncated else ""
    return f"{header}\n{text}{note}"


# --- search ---------------------------------------------------------------


class _ResultParser(HTMLParser):
    """Pulls (title, url, snippet) out of DuckDuckGo's HTML results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._field: str | None = None

    def handle_starttag(self, tag, attrs):
        classes = (dict(attrs).get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append({"title": "", "url": _unwrap_ddg(dict(attrs).get("href", "")), "snippet": ""})
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag in ("a", "td", "div"):
            self._field = None

    def handle_data(self, data):
        if self._field and self.results:
            self.results[-1][self._field] += data


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo links go through a redirect (//duckduckgo.com/l/?uddg=...)."""
    parsed = urlparse(href if "//" not in href[:2] else "https:" + href)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def parse_results(html: str) -> list[dict]:
    parser = _ResultParser()
    parser.feed(html)
    results = []
    for r in parser.results:
        title, snippet = " ".join(r["title"].split()), " ".join(r["snippet"].split())
        if title and r["url"].startswith("http"):
            results.append({"title": title, "url": r["url"], "snippet": snippet})
    return results[:MAX_RESULTS]


def web_search(query: str) -> str:
    """Top results for `query` as a numbered list of title, URL and snippet."""
    query = query.strip()
    if not query:
        raise WebError("empty search query")
    try:
        with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
            resp = client.post(SEARCH_URL, data={"q": query})
    except httpx.HTTPError as exc:
        raise WebError(f"search failed: {exc}") from exc

    results = parse_results(resp.text) if resp.status_code == 200 else []
    if not results:
        return (
            f"No search results for {query!r} (the search service may be rate-limiting; "
            "try again with different wording, or use fetch_url on a URL you already know)."
        )
    lines = [f"Search results for {query!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}" + (f"\n   {r['snippet']}" if r["snippet"] else ""))
    return "\n".join(lines)
