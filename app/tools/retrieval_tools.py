"""Search arXiv and return paper metadata as JSON.

The public arXiv Atom API is the preferred source because it returns structured
XML. When that API is temporarily rate-limited or slow, this module falls back
to the normal arxiv.org search page, first through `cdp` browser automation and
then through a direct HTML request. All paths are normalized into the same JSON
shape so the rest of the agent/RAG project can consume results consistently.
"""

import argparse
import json
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from html import unescape
from typing import Any
from urllib.parse import urlencode

import requests


# arXiv exposes two useful public surfaces:
# 1. export.arxiv.org/api/query -> official Atom XML API.
# 2. arxiv.org/search/ -> human-facing HTML search page used as a fallback.
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_WEB_SEARCH_URL = "https://arxiv.org/search/"

# Include contact information because arXiv asks API clients to identify
# themselves. This also makes traffic easier to diagnose if arXiv throttles us.
ARXIV_USER_AGENT = "explainable-agentic-rag/0.1 contact: neeraj1909@gmail.com"

# requests accepts timeout as (connect_timeout, read_timeout). Keep the connect
# timeout short, but allow enough read time for arXiv's sometimes-slow responses.
REQUEST_TIMEOUT = (5, 30)


class ArxivApiUnavailable(Exception):
    """Raised when the primary/fallback arXiv path cannot return results."""


def _clean_html(raw_html: str) -> str:
    """Convert a small arXiv HTML fragment to compact plain text."""

    without_tags = re.sub(r"<[^>]+>", " ", raw_html)
    return " ".join(unescape(without_tags).split())


def _first_match(pattern: str, text: str, default: str = "") -> str:
    """Return the first regex capture group or a safe default.

    The HTML fallback parser is intentionally dependency-free, so small helper
    functions keep regex extraction readable and defensive.
    """

    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else default


def _parse_web_date(raw_date: str) -> str:
    """Parse arXiv web dates such as '13 May, 2026' when possible."""

    raw_date = raw_date.strip()
    try:
        return datetime.strptime(raw_date, "%d %B, %Y").date().isoformat()
    except ValueError:
        return raw_date


def _parse_atom_feed(feed_xml: str) -> list[dict]:
    """Convert arXiv Atom XML into the project-level paper schema."""

    root = ET.fromstring(feed_xml)

    # Atom XML is namespace-qualified, so every lookup must include the namespace
    # map. Without this, ElementTree will not find fields like <entry> or <title>.
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []

    # Each <entry> is one paper. Normalize whitespace because arXiv often wraps
    # titles and abstracts across multiple XML lines.
    for entry in root.findall("atom:entry", ns):
        title = " ".join(entry.findtext("atom:title", "", ns).split())
        abstract = " ".join(entry.findtext("atom:summary", "", ns).split())
        arxiv_url = entry.findtext("atom:id", "", ns)

        authors = [
            author.findtext("atom:name", "", ns).strip()
            for author in entry.findall("atom:author", ns)
        ]

        # arXiv includes multiple links per paper. The PDF link is identified by
        # title="pdf" rather than by position, so search all links explicitly.
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

        categories = [
            category.attrib.get("term")
            for category in entry.findall("atom:category", ns)
            if category.attrib.get("term")
        ]

        papers.append(
            {
                "title": title,
                "authors": authors,
                "published": entry.findtext("atom:published", "", ns),
                "abstract": abstract,
                "arxiv_url": arxiv_url,
                "pdf_url": pdf_url,
                "categories": categories,
            }
        )

    return papers


def _search_arxiv_api(query: str, max_results: int, sort_by: str) -> list[dict]:
    """Search the official arXiv Atom API and parse its XML response."""

    # Keep the query inside an all:"..." arXiv search expression. User-provided
    # double quotes would break that expression, so replace them with spaces.
    safe_query = query.replace('"', " ")
    params = {
        "search_query": f'all:"{safe_query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    headers = {"User-Agent": ARXIV_USER_AGENT}
    response = requests.get(
        f"{ARXIV_API_URL}?{urlencode(params)}",
        timeout=REQUEST_TIMEOUT,
        headers=headers,
    )

    # Do not sleep/retry here. The public search function will immediately try
    # fallbacks so users do not stare at repeated "rate limited" messages.
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        detail = "arXiv Atom API returned HTTP 429"
        if retry_after:
            detail += f"; Retry-After={retry_after}s"
        raise ArxivApiUnavailable(detail)

    if response.status_code in (500, 502, 503, 504):
        raise ArxivApiUnavailable(
            f"arXiv Atom API returned HTTP {response.status_code}"
        )

    response.raise_for_status()
    return _parse_atom_feed(response.text)


def _parse_arxiv_search_html(html: str, max_results: int) -> list[dict]:
    """Parse arxiv.org search-result HTML into the same paper schema.

    This fallback parser intentionally uses the standard library instead of
    BeautifulSoup so the script keeps working with only the existing project
    dependencies.
    """

    # arXiv renders each search hit as <li class="arxiv-result">. Capture each
    # block first, then extract title/authors/links/categories from that block.
    result_blocks = re.findall(
        r'<li class="arxiv-result">(.*?)(?=<li class="arxiv-result">|</ol>)',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    papers = []
    for block in result_blocks[:max_results]:
        arxiv_url = _first_match(
            r'<p class="list-title[^>]*>\s*<a href="([^"]+/abs/[^"]+)">',
            block,
        )
        pdf_url = _first_match(r'<a href="([^"]+/pdf/[^"]+)">\s*pdf\s*</a>', block)

        title_html = _first_match(r'<p class="title[^"]*"[^>]*>(.*?)</p>', block)
        title = _clean_html(title_html)

        authors_html = _first_match(r'<p class="authors"[^>]*>(.*?)</p>', block)
        authors = [
            _clean_html(author)
            for author in re.findall(
                r'<a [^>]*>(.*?)</a>', authors_html, flags=re.DOTALL
            )
        ]

        # Prefer the hidden full abstract. If arXiv changes the page or only a
        # short abstract is present, fall back to whatever abstract text exists.
        abstract_html = _first_match(
            r'<span class="abstract-full[^"]*"[^>]*>(.*?)<a class="is-size-7"',
            block,
        ) or _first_match(
            r'<p class="abstract[^"]*"[^>]*>(.*?)</p>',
            block,
        )
        abstract = re.sub(r"^Abstract\s*:\s*", "", _clean_html(abstract_html))

        submitted_html = _first_match(r'<p class="is-size-7"[^>]*>(.*?)</p>', block)
        submitted_text = _clean_html(submitted_html)
        submitted_date = _first_match(r"Submitted\s+([^;]+)", submitted_text)

        categories = [
            _clean_html(category)
            for category in re.findall(
                r'<span class="tag[^"]*"[^>]*>(.*?)</span>',
                block,
                flags=re.DOTALL,
            )
        ]

        if title or arxiv_url:
            papers.append(
                {
                    "title": title,
                    "authors": authors,
                    "published": (
                        _parse_web_date(submitted_date) if submitted_date else ""
                    ),
                    "abstract": abstract,
                    "arxiv_url": arxiv_url,
                    "pdf_url": pdf_url or None,
                    "categories": categories,
                }
            )

    return papers


def _arxiv_web_search_url(query: str, sort_by: str) -> str:
    """Build the human-facing arxiv.org search URL used by fallbacks."""

    # The human/browser search page accepts only fixed page sizes. Fetch one small
    # page and then trim to the tool's max_results contract.
    order = "-submitted_date" if sort_by == "submittedDate" else ""
    params = {
        "query": query,
        "searchtype": "all",
        "abstracts": "show",
        "order": order,
        "size": 25,
    }
    return f"{ARXIV_WEB_SEARCH_URL}?{urlencode(params)}"


def _run_cdp_json(args: list[str], timeout_seconds: int = 30) -> dict:
    """Run a cdp CLI command and return parsed JSON output.

    cdp is used only as a fallback when the Atom API is unavailable. Centralizing
    subprocess handling keeps the browser fallback readable and makes all cdp
    failures look like normal arXiv fallback failures to the public function.
    """

    if not shutil.which("cdp"):
        raise ArxivApiUnavailable("cdp CLI is not installed")

    try:
        # Always request JSON from cdp so callers can validate success without
        # scraping terminal text.
        completed = subprocess.run(
            ["cdp", *args, "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ArxivApiUnavailable(
            f"cdp command timed out: cdp {' '.join(args)}"
        ) from exc

    output = completed.stdout.strip()
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ArxivApiUnavailable(
            f"cdp command failed: cdp {' '.join(args)}; {detail}"
        )

    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ArxivApiUnavailable(f"cdp returned non-JSON output: {output[:200]}") from exc

    if isinstance(data, dict) and data.get("ok") is False:
        message = data.get("message") or data.get("error") or data.get("code") or data
        raise ArxivApiUnavailable(f"cdp command returned ok=false: {message}")

    return data


def _search_arxiv_cdp(query: str, max_results: int, sort_by: str) -> list[dict]:
    """Use a real browser via cdp to fetch arxiv.org search results.

    This is useful when direct Python requests are being throttled but the normal
    browser page still loads. The page HTML is parsed with the same HTML parser
    used by the direct web fallback.
    """

    url = _arxiv_web_search_url(query, sort_by)
    target_id = None

    try:
        opened = _run_cdp_json(["open", url], timeout_seconds=20)
        target_id = (opened.get("page") or {}).get("id")
        if not target_id:
            raise ArxivApiUnavailable("cdp open did not return a page target id")

        # cdp --help shows first-class wait/html commands. Use them instead of
        # hand-written browser automation so this remains shell/debuggable.
        try:
            _run_cdp_json(
                [
                    "wait",
                    "selector",
                    "li.arxiv-result",
                    "--target",
                    target_id,
                    "--timeout",
                    "25s",
                ],
                timeout_seconds=35,
            )
        except ArxivApiUnavailable:
            # Still inspect the page HTML below; a no-results page legitimately
            # has no result selector, while a rate-limit page gives a useful
            # diagnostic.
            pass

        html_result = _run_cdp_json(
            [
                "html",
                "body",
                "--target",
                target_id,
                "--max-chars",
                "0",
                "--limit",
                "1",
            ],
            timeout_seconds=35,
        )
        items = (html_result.get("html") or {}).get("items") or []
        html = items[0].get("html", "") if items else ""
        papers = _parse_arxiv_search_html(html, max_results)
        if not papers:
            page_summary = _clean_html(html)[:200]
            raise ArxivApiUnavailable(
                f"cdp arXiv search returned no parsed papers: {page_summary}"
            )

        return papers
    finally:
        # The browser fallback opens a temporary tab. Close it so repeated tool
        # calls do not leak tabs or push the cdp daemon over its resource budget.
        if target_id:
            try:
                _run_cdp_json(
                    ["page", "close", "--target", target_id], timeout_seconds=10
                )
            except ArxivApiUnavailable:
                pass


def _search_arxiv_web(query: str, max_results: int, sort_by: str) -> list[dict]:
    """Fetch and parse arxiv.org search HTML with requests as a last fallback."""

    headers = {
        "User-Agent": f"Mozilla/5.0 ({ARXIV_USER_AGENT})",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    response = requests.get(
        _arxiv_web_search_url(query, sort_by),
        timeout=REQUEST_TIMEOUT,
        headers=headers,
    )
    response.raise_for_status()
    return _parse_arxiv_search_html(response.text, max_results)


def _results_payload(
    query: str,
    papers: list[dict],
    retrieval_method: str,
    warning: str | None = None,
) -> str:
    """Build the successful JSON response returned by search_papers()."""

    payload = {
        "query": query,
        "source": "arXiv",
        "retrieval_method": retrieval_method,
        "results": papers,
    }
    if warning:
        payload["warning"] = warning

    return json.dumps(payload, indent=2)


def _error_payload(query: str, message: str) -> str:
    """Build a JSON error response while preserving the normal result shape."""

    return json.dumps(
        {
            "query": query,
            "source": "arXiv",
            "error": message,
            "results": [],
        },
        indent=2,
    )


def search_papers(query: str, max_results: int = 10):
    """
    Search arXiv for papers relevant to a research query.

    Returns JSON containing paper title, authors, publication date, abstract,
    arXiv URL, PDF URL, and categories.

    The official arXiv Atom API is preferred. If it is temporarily rate limited
    or unavailable, fall back to cdp/browser-backed arxiv.org search and then a
    direct arxiv.org search request instead of retrying the API in a noisy loop.
    """

    # Keep the public API forgiving: trim accidental spaces and cap result count
    # so a single tool call cannot ask arXiv for a very large page.
    query = query.strip()
    max_results = max(1, min(max_results, 10))

    # Simple intent detection: if the query asks for recent/latest/new papers,
    # sort by submission date; otherwise let arXiv relevance ranking decide.
    sort_by = "submittedDate" if any(
        keyword in query.lower() for keyword in ["recent", "latest", "new"]
    ) else "relevance"

    # Preferred path: official structured Atom API.
    try:
        papers = _search_arxiv_api(query, max_results, sort_by)
        if papers:
            return _results_payload(query, papers, retrieval_method="atom_api")
        fallback_warning = "arXiv Atom API returned zero results. Used fallback search."
    except (ArxivApiUnavailable, requests.exceptions.Timeout) as exc:
        fallback_warning = f"{exc}. Used fallback search."
    except requests.exceptions.RequestException as exc:
        fallback_warning = f"arXiv Atom API request failed: {exc}. Used fallback search."
    except ET.ParseError as exc:
        fallback_warning = f"arXiv Atom API returned invalid XML: {exc}. Used fallback search."

    # First fallback: load the human arXiv search page in Chrome through cdp.
    try:
        # A tiny courtesy pause avoids immediately following a rejected API
        # request with another arXiv request in the same second.
        time.sleep(1)
        papers = _search_arxiv_cdp(query, max_results, sort_by)
        return _results_payload(
            query,
            papers,
            retrieval_method="cdp_browser_fallback",
            warning=fallback_warning,
        )
    except ArxivApiUnavailable as exc:
        fallback_warning = f"{fallback_warning} cdp browser fallback failed: {exc}."

    # Final fallback: direct HTML request to arxiv.org/search/.
    try:
        time.sleep(1)
        papers = _search_arxiv_web(query, max_results, sort_by)
        return _results_payload(
            query,
            papers,
            retrieval_method="web_search_fallback",
            warning=fallback_warning,
        )
    except requests.exceptions.RequestException as exc:
        return _error_payload(
            query,
            "arXiv API, cdp browser fallback, and web search are unavailable. "
            f"{fallback_warning} Direct web search failed: {exc}",
        )


def summarize_claim(search_result: dict[str, Any], llm_client) -> str:
    """Summarize arXiv search results into an evidence-grounded Markdown brief."""

    query = search_result.get("query", "Unknown query")
    source = search_result.get("source", "Unknown source")
    retrieval_method = search_result.get("retrieval_method", "Unknown method")
    papers = search_result.get("results", [])

    if not papers:
        return f"No papers found for query: {query}"

    categories: Counter[str] = Counter()
    years: list[int] = []

    for paper in papers:
        categories.update(paper.get("categories", []))
        year = extract_year(paper.get("published"))
        if year is not None:
            years.append(year)

    summary_lines = [
        "# Search Summary",
        "",
        f"**Query:** {query}",
        f"**Source:** {source}",
        f"**Retrieval Method:** {retrieval_method}",
        f"**Total Papers Found:** {len(papers)}",
    ]

    if years:
        summary_lines.append(f"**Publication Year Range:** {min(years)}-{max(years)}")

    if categories:
        top_categories = ", ".join(
            f"{category} ({count})" for category, count in categories.most_common(5)
        )
        summary_lines.append(f"**Top Categories:** {top_categories}")

    summary_lines.extend([
        "",
        "## Overall Takeaway",
        generate_overall_takeaway(query, papers, llm_client),
        "",
        "## Papers",
        "",
    ])

    for index, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors", [])
        year = extract_year(paper.get("published"))
        abstract = paper.get("abstract", "")
        arxiv_url = paper.get("arxiv_url", "")

        summary_lines.extend([
            f"### {index}. {title}",
            "",
            f"- **Year:** {year or 'Unknown'}",
            f"- **Authors:** {format_authors(authors)}",
            f"- **Summary:** {summarize_abstract(abstract, llm_client)}",
            f"- **URL:** {arxiv_url}",
            "",
        ])

    return "\n".join(summary_lines)


def call_llm(llm_client, prompt: str) -> str:
    """Invoke the configured chat model and return plain text content."""

    response = llm_client.invoke(prompt)
    return response.content.strip()


def extract_year(published: str | None) -> int | None:
    """Extract an ISO timestamp year when available."""

    if not published:
        return None

    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).year
    except ValueError:
        return None


def format_authors(authors: list[str], max_authors: int = 3) -> str:
    """Format an author list for CLI/agent summaries."""

    if not authors:
        return "Unknown"
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + ", et al."


def generate_overall_takeaway(
    user_query: str,
    paper_summaries: list[dict],
    llm_client,
) -> str:
    """Generate a short evidence-constrained synthesis across retrieved papers."""

    prompt = f"""
Given the following search query and paper summaries, provide an overall takeaway
that captures the main insights and trends related to the query.

User searched for: {user_query}
Retrieved papers: {json.dumps(paper_summaries)}

Write an overall takeaway in 3-5 sentences.
Mention:
- Common themes across the papers
- Dominant methodologies or approaches
- Notable gaps or future directions
- What the user should read first

Do not invent claims beyond the provided summaries.
"""
    return call_llm(llm_client, prompt)


def summarize_abstract(abstract: str, llm_client) -> str:
    """Summarize one abstract without adding unsupported information."""

    prompt = f"""
You are summarizing an academic abstract for a literature search.

Abstract: {abstract}

Return a concise summary in 2-3 sentences. Do not include information that is
not present in the abstract.
"""
    return call_llm(llm_client, prompt)


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter for future non-LLM summary fallbacks."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def main() -> None:
    """Manual CLI for the retrieval/summarization tool module."""

    parser = argparse.ArgumentParser(description="Search and summarize arXiv papers.")
    parser.add_argument("query", nargs="+", help="User search query")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--summary", action="store_true", help="Print summarized output")
    args = parser.parse_args()

    from app.config import get_llm_client

    query = " ".join(args.query)
    raw_result = search_papers(query, max_results=args.max_results)
    if args.summary:
        payload = json.loads(raw_result)
        print(summarize_claim(payload, get_llm_client()))
    else:
        print(raw_result)


if __name__ == "__main__":
    main()
