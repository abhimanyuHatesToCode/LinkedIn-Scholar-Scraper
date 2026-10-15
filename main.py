"""
main.py
========
FastAPI application — LinkedIn + Google Scholar scraper API.

Endpoints:
  GET  /health                → liveness check
  GET  /session/status        → check LinkedIn session
  POST /scrape                → scrape LinkedIn profile (with time_window)
  POST /scrape/scholar        → scrape Google Scholar profile (independent)

Run:
  uvicorn main:app --port 8000     <- NO --reload on Windows
"""

from __future__ import annotations

import asyncio
import sys

# Windows event loop fix — must be set before uvicorn touches the loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

from scraper.linkedin_browser import LinkedInBrowser
from scraper.linkedin_scraper import LinkedInScraper, build_profile_url
from scraper.scholar_scraper import ScholarScraper
from extraction.ai_extractor import AIExtractor
from schemas.faculty_schema import (
    ScrapeRequest,
    ScrapeResponse,
    FacultyProfile,
    TimeWindow,
)
from schemas.scholar_schema import (
    ScholarScrapeRequest,
    ScholarScrapeResponse,
    ScholarProfile,
)


# ── Shared state ──────────────────────────────────────────────────────────────

_browser:        LinkedInBrowser | None = None
_extractor:      AIExtractor | None     = None
_li_scraper:     LinkedInScraper | None = None
_scholar_scraper: ScholarScraper | None = None

# Single lock — Scholar and LinkedIn share the same Chrome instance,
# so we prevent them from running concurrently.
_scrape_lock = asyncio.Lock()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _browser, _extractor, _li_scraper, _scholar_scraper

    print("[startup] Initialising browser and extractors...")
    _browser          = LinkedInBrowser()
    _extractor        = AIExtractor()

    await _browser.start()
    logged_in = await _browser.ensure_logged_in()
    if not logged_in:
        print("[startup] WARNING: LinkedIn session not verified. Run first_login.py.")

    # Both scrapers share the same browser instance
    _li_scraper      = LinkedInScraper(_browser, _extractor)
    _scholar_scraper = ScholarScraper(_browser)

    print("[startup] Ready.")
    yield

    print("[shutdown] Closing browser...")
    await _browser.close()
    print("[shutdown] Done.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Faculty Scraper API",
    description=(
        "Scrapes LinkedIn profiles (identity, publications, projects, "
        "achievements, events) and Google Scholar profiles (metrics, "
        "publications, co-authors, citation graph) independently."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_li_scraper() -> LinkedInScraper:
    if _li_scraper is None:
        raise HTTPException(status_code=503, detail="LinkedIn scraper not initialised.")
    return _li_scraper


def get_scholar_scraper() -> ScholarScraper:
    if _scholar_scraper is None:
        raise HTTPException(status_code=503, detail="Scholar scraper not initialised.")
    return _scholar_scraper


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "3.0.0"}


@app.get("/session/status", tags=["meta"])
async def session_status():
    """Check whether the browser has an active LinkedIn session."""
    if _browser is None:
        return JSONResponse({"logged_in": False, "reason": "browser_not_started"})

    from scraper.linkedin_browser import LINKEDIN_HOME, _is_logged_in
    page = await _browser.new_page()
    try:
        await page.goto(LINKEDIN_HOME, wait_until="domcontentloaded")
        logged_in = await _is_logged_in(page)
        return {"logged_in": logged_in}
    except Exception as exc:
        return JSONResponse({"logged_in": False, "reason": str(exc)})
    finally:
        await page.close()


@app.post("/scrape", response_model=ScrapeResponse, tags=["linkedin"])
async def scrape_linkedin(
    body:    ScrapeRequest,
    scraper: Annotated[LinkedInScraper, Depends(get_li_scraper)],
) -> ScrapeResponse:
    """
    Scrape a LinkedIn profile.

    Body:
      profile      — LinkedIn slug or full URL (own profile / 1st-degree connection)
      time_window  — 1, 3, or 6 months (default: 3)

    Returns:
      FacultyProfile with identity, publications, projects, achievements, events.
    """
    profile_url = build_profile_url(body.profile)

    async with _scrape_lock:
        try:
            profile: FacultyProfile = await scraper.scrape(
                body.profile,
                time_window=body.time_window,
            )
            return ScrapeResponse(success=True, profile_url=profile_url, data=profile)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@app.post("/scrape/scholar", response_model=ScholarScrapeResponse, tags=["scholar"])
async def scrape_scholar(
    body:    ScholarScrapeRequest,
    scraper: Annotated[ScholarScraper, Depends(get_scholar_scraper)],
) -> ScholarScrapeResponse:
    """
    Scrape a Google Scholar profile — completely independent from LinkedIn.

    Body:
      url — Full Google Scholar profile URL
            e.g. https://scholar.google.com/citations?user=ABC123&hl=en

    Returns:
      ScholarProfile with:
        - name, affiliation, verified_email, research_interests
        - metrics (citations, h-index, i10-index — all-time + since 2019)
        - citation_graph (year-by-year citation counts)
        - publications (up to 20, with citation counts and Scholar URLs)
        - co_authors (sidebar entries with affiliation and profile links)

    Notes:
      - Does NOT require LinkedIn authentication
      - Uses the same Chrome instance as LinkedIn scraping
      - Will raise a 429-style error if Google Scholar serves a CAPTCHA
        (wait a few minutes and retry if this happens)
    """
    async with _scrape_lock:
        try:
            profile: ScholarProfile = await scraper.scrape(body.url)
            return ScholarScrapeResponse(
                success=True,
                profile_url=body.url,
                data=profile,
            )
        except ValueError as exc:
            # CAPTCHA or invalid URL
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop="asyncio",
    )
