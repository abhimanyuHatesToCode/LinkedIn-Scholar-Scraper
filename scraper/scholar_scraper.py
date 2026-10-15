"""
scholar_scraper.py
===================
Scrapes a Google Scholar profile page using the existing LinkedInBrowser
(opens a Scholar tab in the already-running Chrome instance) and
BeautifulSoup for DOM parsing.

Why BeautifulSoup here (unlike LinkedIn):
  Scholar's HTML is server-rendered by Google and has been structurally
  stable for years. The CSS selectors used here (#gsc_prf_in, .gsc_rsb_std,
  .gsc_a_tr etc.) are part of Google's Scholar template and change far less
  frequently than LinkedIn's React class names. Direct DOM parsing is faster,
  cheaper (no LLM tokens), and more accurate for structured tabular data.

Why no separate browser:
  The LinkedInBrowser already holds a warm, authenticated Chrome instance.
  Opening a Scholar tab inside it is natural browsing behaviour and avoids
  the overhead of cold-starting a second browser process.

Rate limiting protection:
  Scholar returns a CAPTCHA HTML page (not an HTTP error) when hit too fast.
  We detect this and raise clearly rather than returning garbage data.
  Random delays between the page load and parsing mimic human reading pace.

Publication cap: 20 (project design decision).
  Scholar paginates in blocks of 20 by default. We request pagesize=20
  directly in the URL so we always get exactly one page — no pagination
  logic needed, no extra requests.

Citation graph:
  The bar chart data is embedded as a JavaScript array literal in a
  <script> tag on the page. We extract it with a regex — no JS execution
  needed, no extra requests.
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

from bs4 import BeautifulSoup
from patchright.async_api import Page

from scraper.linkedin_browser import LinkedInBrowser
from schemas.scholar_schema import (
    ScholarProfile,
    ScholarMetrics,
    CitationYear,
    ScholarPublication,
    CoAuthor,
)

SCHOLAR_BASE = "https://scholar.google.com"
PUB_CAP      = 20  # max publications to return

# Scholar returns a CAPTCHA page when bot-detected.
# These strings appear in the CAPTCHA HTML body.
_CAPTCHA_SIGNALS = [
    "please show you're not a robot",
    "our systems have detected unusual traffic",
    "unusual traffic from your computer",
    "captcha",
]


# ── ScholarScraper ────────────────────────────────────────────────────────────

class ScholarScraper:
    """
    Scrapes a Google Scholar profile and returns a validated ScholarProfile.

    Usage:
        scraper = ScholarScraper(browser)
        profile = await scraper.scrape("https://scholar.google.com/citations?user=ABC123")
    """

    def __init__(self, browser: LinkedInBrowser) -> None:
        self.browser = browser

    async def scrape(self, url: str) -> ScholarProfile:
        """
        Full scrape pipeline:
          1. Normalise URL  (force pagesize=20, hl=en)
          2. Open Scholar tab in existing Chrome context
          3. Fetch rendered HTML
          4. Parse with BeautifulSoup
          5. Extract all fields
          6. Return validated ScholarProfile

        Args:
            url: Full Google Scholar profile URL.

        Returns:
            Validated ScholarProfile.

        Raises:
            ValueError  on CAPTCHA detection or invalid URL.
            RuntimeError on unexpected parse failures.
        """
        canonical = _normalise_url(url, pagesize=PUB_CAP)
        print(f"\n[scholar] ── Scraping: {canonical} ──")

        page: Page = await self.browser.new_page()
        try:
            html = await self._fetch_html(page, canonical)
            _check_for_captcha(html, canonical)
            profile = _parse_profile(html, canonical)
            print(f"[scholar] ✓ Done — {profile.display()}")
            return profile
        finally:
            await page.close()

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def _fetch_html(self, page: Page, url: str) -> str:
        """Navigate to Scholar URL and return the fully rendered HTML."""
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        # Random pause — mimics human reading time, reduces CAPTCHA risk
        await asyncio.sleep(random.uniform(2.5, 4.0))

        html = await page.content()
        print(f"[scholar] Fetched {len(html):,} bytes of HTML.")
        return html


# ── URL normalisation ─────────────────────────────────────────────────────────

def _normalise_url(url: str, pagesize: int = 20) -> str:
    """
    Ensure the Scholar URL:
      - Has hl=en  (English UI — selectors depend on English labels)
      - Has pagesize=N  (controls how many publications load on first page)
      - Strips the cstart parameter (always start from the beginning)

    Accepts both:
      https://scholar.google.com/citations?user=ABC123
      https://scholar.google.com/citations?user=ABC123&hl=en&pagesize=100
    """
    parsed   = urlparse(url)
    params   = parse_qs(parsed.query, keep_blank_values=True)

    # Force English and our desired pagesize
    params["hl"]       = ["en"]
    params["pagesize"] = [str(pagesize)]
    params.pop("cstart", None)

    # Flatten the lists that parse_qs creates
    flat = {k: v[0] for k, v in params.items()}

    new_query = urlencode(flat)
    return urlunparse(parsed._replace(query=new_query))


# ── HTML parsing ──────────────────────────────────────────────────────────────

def _parse_profile(html: str, url: str) -> ScholarProfile:
    """
    Parse the full Scholar profile HTML into a ScholarProfile.
    Each sub-parser is isolated — a failure in one section never
    crashes the others.
    """
    soup = BeautifulSoup(html, "lxml")

    name        = _parse_name(soup)
    affiliation = _parse_affiliation(soup)
    email       = _parse_verified_email(soup)
    interests   = _parse_interests(soup)
    metrics     = _parse_metrics(soup)
    cit_graph   = _parse_citation_graph(html)   # uses raw HTML for regex
    publications = _parse_publications(soup)
    co_authors  = _parse_co_authors(soup)

    return ScholarProfile(
        name               = name,
        affiliation        = affiliation,
        verified_email     = email,
        research_interests = interests,
        scholar_url        = url,
        metrics            = metrics,
        citation_graph     = cit_graph,
        publications       = publications,
        co_authors         = co_authors,
    )


# ── Section parsers ───────────────────────────────────────────────────────────

def _parse_name(soup: BeautifulSoup) -> Optional[str]:
    """#gsc_prf_in — the large name heading on the profile."""
    tag = soup.select_one("#gsc_prf_in")
    return tag.get_text(strip=True) if tag else None


def _parse_affiliation(soup: BeautifulSoup) -> Optional[str]:
    """
    .gsc_prf_il — the first item under the name is affiliation.
    Scholar renders multiple .gsc_prf_il elements (affiliation, email, homepage).
    The first one is always the institutional affiliation.
    """
    tags = soup.select(".gsc_prf_il")
    for tag in tags:
        text = tag.get_text(strip=True)
        # Skip email-looking and URL-looking lines
        if "@" not in text and "http" not in text and text:
            return text
    return None


def _parse_verified_email(soup: BeautifulSoup) -> Optional[str]:
    """
    The 'Verified email at domain.edu' line appears in a .gsc_prf_il
    element that contains an anchor tag pointing to the domain.
    """
    for tag in soup.select(".gsc_prf_il"):
        text = tag.get_text(strip=True)
        if "verified email" in text.lower():
            return text
    return None


def _parse_interests(soup: BeautifulSoup) -> list[str]:
    """
    .gsc_prf_inta — each clickable research interest tag.
    Returns a list of interest strings.
    """
    tags = soup.select(".gsc_prf_inta")
    return [t.get_text(strip=True) for t in tags if t.get_text(strip=True)]


def _parse_metrics(soup: BeautifulSoup) -> Optional[ScholarMetrics]:
    """
    .gsc_rsb_std — the metrics table cells.
    Scholar renders the table as:
      Row 0 header | "All" | "Since 2019"
      Row 1: Citations | val | val
      Row 2: h-index   | val | val
      Row 3: i10-index | val | val

    The .gsc_rsb_std cells appear in reading order:
      [0] citations_all  [1] citations_since
      [2] h_index_all    [3] h_index_since
      [4] i10_all        [5] i10_since
    """
    cells = soup.select(".gsc_rsb_std")
    if not cells:
        return None

    def _int(idx: int) -> Optional[int]:
        if idx >= len(cells):
            return None
        text = cells[idx].get_text(strip=True)
        return int(text) if text.isdigit() else None

    return ScholarMetrics(
        citations_total      = _int(0),
        citations_since_2019 = _int(1),
        h_index_total        = _int(2),
        h_index_since_2019   = _int(3),
        i10_index_total      = _int(4),
        i10_index_since_2019 = _int(5),
    )


def _parse_citation_graph(html: str) -> list[CitationYear]:
    """
    The citation bar chart data is embedded as a JavaScript object literal
    inside a <script> tag. It looks like:

        {"graph":[{"year":2015,"citations":12},{"year":2016,"citations":34},...]}

    We extract it with a regex — no JS execution needed.
    This is more reliable than trying to find a DOM element for the chart.
    """
    # Match the graph array inside the Scholar page script
    match = re.search(
        r'"graph"\s*:\s*(\[\s*\{.*?\}\s*\])',
        html,
        re.DOTALL,
    )
    if not match:
        return []

    import json
    try:
        raw_graph = json.loads(match.group(1))
        result = []
        for entry in raw_graph:
            year = entry.get("year")
            cits = entry.get("citations", 0)
            if isinstance(year, int) and isinstance(cits, int):
                result.append(CitationYear(year=year, citations=cits))
        return result
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def _parse_publications(soup: BeautifulSoup) -> list[ScholarPublication]:
    """
    .gsc_a_tr — each row in the publications table.

    Each row contains:
      .gsc_a_at   — title (also an anchor to the Scholar detail page)
      .gsc_a_at + siblings — authors and venue are in the grey text below the title
      .gsc_a_y span — publication year
      .gsc_a_c  — citation count (may be empty, '*', or a number)

    Authors and venue are in .gs_gray divs inside the title cell.
    The first .gs_gray is the authors string.
    The second .gs_gray is the venue string.
    """
    rows = soup.select(".gsc_a_tr")[:PUB_CAP]
    publications = []

    for row in rows:
        try:
            # Title + URL
            title_tag = row.select_one(".gsc_a_at")
            title     = title_tag.get_text(strip=True) if title_tag else None
            if not title:
                continue

            href = title_tag.get("href", "") if title_tag else ""
            pub_url = (SCHOLAR_BASE + href) if href.startswith("/") else (href or None)

            # Authors and venue (the two grey lines below the title)
            gray_tags = row.select(".gs_gray")
            authors   = gray_tags[0].get_text(strip=True) if len(gray_tags) > 0 else None
            venue     = gray_tags[1].get_text(strip=True) if len(gray_tags) > 1 else None

            # Year
            year_tag  = row.select_one(".gsc_a_y span")
            year_text = year_tag.get_text(strip=True) if year_tag else ""
            year      = int(year_text) if year_text.isdigit() else None

            # Citations
            cite_tag  = row.select_one(".gsc_a_c")
            cite_text = cite_tag.get_text(strip=True) if cite_tag else ""

            publications.append(ScholarPublication(
                title       = title,
                authors     = authors,
                venue       = venue,
                year        = year,
                citations   = cite_text,   # coerce_citations validator handles this
                scholar_url = pub_url,
            ))
        except Exception as exc:
            print(f"[scholar] Skipping publication row: {exc}")
            continue

    return publications


def _parse_co_authors(soup: BeautifulSoup) -> list[CoAuthor]:
    """
    .gsc_rsb_aa — each co-author card in the sidebar.

    Each card contains:
      .gsc_rsb_aa img or a — link to the co-author's Scholar profile
      .gsc_rsb_name a      — name text
      .gsc_rsb_aff         — affiliation text (may be absent)
    """
    cards = soup.select(".gsc_rsb_aa")
    co_authors = []

    for card in cards:
        try:
            name_tag    = card.select_one(".gsc_rsb_name a") or card.select_one("a")
            name        = name_tag.get_text(strip=True) if name_tag else None
            if not name:
                continue

            href        = name_tag.get("href", "") if name_tag else ""
            profile_url = (SCHOLAR_BASE + href) if href.startswith("/") else (href or None)

            affil_tag   = card.select_one(".gsc_rsb_aff")
            affiliation = affil_tag.get_text(strip=True) if affil_tag else None

            co_authors.append(CoAuthor(
                name        = name,
                affiliation = affiliation,
                scholar_url = profile_url,
            ))
        except Exception as exc:
            print(f"[scholar] Skipping co-author card: {exc}")
            continue

    return co_authors


# ── CAPTCHA detection ─────────────────────────────────────────────────────────

def _check_for_captcha(html: str, url: str) -> None:
    """
    Raise a clear ValueError if Scholar returned a CAPTCHA challenge page
    instead of the actual profile. This prevents garbage data from being
    returned to the caller.

    Google Scholar does NOT use HTTP error codes for CAPTCHA — it returns
    HTTP 200 with a challenge HTML body. So we must check the content.
    """
    lower = html.lower()
    for signal in _CAPTCHA_SIGNALS:
        if signal in lower:
            raise ValueError(
                f"[scholar] Google Scholar CAPTCHA detected for {url!r}. "
                "Wait a few minutes before retrying, or restart the browser session."
            )
#   https://scholar.google.com/citations?user=VjJm3zYAAAAJ&hl=en