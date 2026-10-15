"""
linkedin_scraper.py
====================
Orchestrates the full two-phase scraping pipeline:

  Phase 1 — Profile page
    Navigates to /in/{slug}, scrolls the full page, extracts visible text,
    calls LLM for: identity, publications, projects, achievements.
    Applies time filter to dated items from profile sections.

  Phase 2 — Activity feed
    Navigates to /in/{slug}/recent-activity/all/, collects posts within
    the time window, calls LLM for: events.

  Merge — Combines both phases into a single FacultyProfile.

Time filtering rules (Option C, agreed in design):
  - Dated item with full date (YYYY-MM) : included only if >= cutoff
  - Dated item with year-only  (YYYY)   : always included
  - Item with no date                   : excluded
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from dateutil.relativedelta import relativedelta
from patchright.async_api import Page

from scraper.linkedin_browser import LinkedInBrowser
from scraper.activity_scraper import ActivityScraper
from extraction.ai_extractor import AIExtractor
from schemas.faculty_schema import (
    FacultyProfile,
    Publication, PublicationType,
    Project,
    Achievement, AchievementType,
    Event, EventType, EventRole,
    TimeWindow,
)

LINKEDIN_BASE = "https://www.linkedin.com"


# ── URL helpers ───────────────────────────────────────────────────────────────

def build_profile_url(identifier: str) -> str:
    identifier = identifier.strip().rstrip("/")
    if identifier.startswith("http"):
        return identifier
    if "linkedin.com/in/" in identifier:
        return "https://" + identifier.lstrip("https://")
    return f"{LINKEDIN_BASE}/in/{identifier}"


def _extract_slug(url: str) -> str:
    match = re.search(r"/in/([^/?#]+)", url)
    return match.group(1) if match else url


# ── LinkedInScraper ───────────────────────────────────────────────────────────

class LinkedInScraper:
    """
    High-level two-phase scraper.

    Usage:
        scraper = LinkedInScraper(browser, extractor)
        profile = await scraper.scrape("williamhgates", time_window=TimeWindow.THREE_MONTHS)
    """

    def __init__(self, browser: LinkedInBrowser, extractor: AIExtractor) -> None:
        self.browser          = browser
        self.extractor        = extractor
        self.activity_scraper = ActivityScraper()

    async def scrape(
        self,
        profile_identifier: str,
        time_window: TimeWindow = TimeWindow.THREE_MONTHS,
    ) -> FacultyProfile:
        """
        Full pipeline: profile page → activity feed → merged FacultyProfile.

        Args:
            profile_identifier: LinkedIn slug or full profile URL.
            time_window: TimeWindow enum value (1, 3, or 6 months).

        Returns:
            Validated FacultyProfile Pydantic model.

        Raises:
            ValueError if auth-wall or profile not found.
        """
        url    = build_profile_url(profile_identifier)
        slug   = _extract_slug(url)
        months = time_window.value
        cutoff = datetime.now() - relativedelta(months=months)

        print(f"\n[scraper] ── Pipeline start: {slug} | window: {months} months ──")

        # ── Phase 1: Profile page ─────────────────────────────────────────────
        profile_data = await self._scrape_profile_page(url, slug, cutoff)

        # ── Phase 2: Activity feed ────────────────────────────────────────────
        events = await self._scrape_events(slug, months)

        # ── Merge into FacultyProfile ─────────────────────────────────────────
        profile = self._build_profile(profile_data, events, months)
        print(f"[scraper] ✓ Done — {profile.total_items} items extracted.")
        return profile

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    async def _scrape_profile_page(
        self,
        url:    str,
        slug:   str,
        cutoff: datetime,
    ) -> dict:
        """
        Navigate to the profile page, scroll it fully, extract text,
        run two LLM calls (identity + sections), apply time filter.
        """
        page: Page = await self.browser.new_page()
        try:
            print(f"[scraper] Phase 1 — fetching profile page")
            visible_text = await self.browser.fetch_page(page, url)
            _check_for_blocks(visible_text, url)

            # LLM call 1: identity (name, title, affiliation, location, summary)
            print("[scraper] LLM call 1/2 — identity")
            identity = await self.extractor.extract_identity(visible_text)

            # LLM call 2: profile sections (publications, projects, achievements)
            print("[scraper] LLM call 2/2 — profile sections")
            sections = await self.extractor.extract_profile_sections(visible_text)

            # Apply time filter to each section
            return {
                "identity": identity,
                "publications": _filter_publications(sections.get("publications", []), cutoff),
                "projects":     _filter_projects(sections.get("projects", []),     cutoff),
                "achievements": _filter_achievements(sections.get("achievements", []), cutoff),
            }
        finally:
            await page.close()

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    async def _scrape_events(self, slug: str, months: int) -> list[dict]:
        """
        Navigate to the activity feed, collect filtered posts,
        run the events LLM call.
        """
        page: Page = await self.browser.new_page()
        try:
            print(f"[scraper] Phase 2 — fetching activity feed")
            raw_posts = await self.activity_scraper.scrape(page, slug, months)

            if not raw_posts:
                print("[scraper] No posts returned — skipping events LLM call.")
                return []

            print(f"[scraper] LLM call — events from {len(raw_posts)} posts")
            return await self.extractor.extract_events(raw_posts)

        except Exception as exc:
            # Activity feed failures are non-fatal — return empty events
            print(f"[scraper] Activity feed error (non-fatal): {exc}")
            return []
        finally:
            await page.close()

    # ── Merge ─────────────────────────────────────────────────────────────────

    def _build_profile(
        self,
        profile_data: dict,
        raw_events:   list[dict],
        months:       int,
    ) -> FacultyProfile:
        """
        Convert raw LLM output dicts into validated Pydantic models
        and assemble the final FacultyProfile.
        """
        identity = profile_data.get("identity", {})

        # Build typed lists — invalid items are silently skipped
        publications = _build_publications(profile_data.get("publications", []))
        projects     = _build_projects(profile_data.get("projects", []))
        achievements = _build_achievements(profile_data.get("achievements", []))
        events       = _build_events(raw_events)

        return FacultyProfile(
            name        = identity.get("name"),
            title       = identity.get("title"),
            affiliation = identity.get("affiliation"),
            location    = identity.get("location"),
            summary     = identity.get("summary"),
            confidence  = float(identity.get("confidence", 0.0)),
            publications = publications,
            projects     = projects,
            achievements = achievements,
            events       = events,
            time_window_months = months,
        )


# ── Time filtering ────────────────────────────────────────────────────────────
# Option C rules:
#   - year-only ("2024")          → always include
#   - full date ("2024-03")       → include only if >= cutoff
#   - None / missing              → exclude

def _date_passes_filter(date_str: Optional[str], cutoff: datetime) -> bool:
    """Return True if the date string passes the time window filter."""
    if date_str is None:
        return False  # No date → exclude

    s = date_str.strip()

    # Year-only → always include regardless of window
    if re.match(r"^\d{4}$", s):
        return True

    # Try to parse YYYY-MM
    match = re.match(r"^(\d{4})-(\d{2})", s)
    if match:
        try:
            item_dt = datetime(int(match.group(1)), int(match.group(2)), 1)
            return item_dt >= cutoff
        except ValueError:
            return True  # Can't parse → include (be generous)

    # Unrecognised format → include
    return True


def _filter_publications(items: list[dict], cutoff: datetime) -> list[dict]:
    return [i for i in items if _date_passes_filter(i.get("date"), cutoff)]


def _filter_projects(items: list[dict], cutoff: datetime) -> list[dict]:
    # Use date_end for projects (when was it active/completed)
    # "Present" is treated as always included
    result = []
    for item in items:
        end = item.get("date_end") or ""
        if end.strip().lower() == "present":
            result.append(item)
        else:
            if _date_passes_filter(end or item.get("date_start"), cutoff):
                result.append(item)
    return result


def _filter_achievements(items: list[dict], cutoff: datetime) -> list[dict]:
    return [i for i in items if _date_passes_filter(i.get("date_issued"), cutoff)]


# ── Pydantic model builders ───────────────────────────────────────────────────
# Silently skip items that fail validation — a partial result is better
# than a crashed scrape.

def _build_publications(raw_list: list[dict]) -> list[Publication]:
    result = []
    for item in raw_list:
        try:
            result.append(Publication(
                title       = item.get("title", "Untitled"),
                type        = _safe_enum(PublicationType, item.get("type"), PublicationType.PAPER),
                publisher   = item.get("publisher"),
                date        = item.get("date"),
                description = item.get("description"),
                url         = item.get("url"),
                confidence  = float(item.get("confidence", 0.0)),
            ))
        except Exception as exc:
            print(f"[scraper] Skipping invalid publication: {exc}")
    return result


def _build_projects(raw_list: list[dict]) -> list[Project]:
    result = []
    for item in raw_list:
        try:
            collabs = item.get("collaborators") or []
            if isinstance(collabs, str):
                collabs = [collabs]
            result.append(Project(
                title         = item.get("title", "Untitled"),
                date_start    = item.get("date_start"),
                date_end      = item.get("date_end"),
                description   = item.get("description"),
                collaborators = collabs,
                url           = item.get("url"),
                confidence    = float(item.get("confidence", 0.0)),
            ))
        except Exception as exc:
            print(f"[scraper] Skipping invalid project: {exc}")
    return result


def _build_achievements(raw_list: list[dict]) -> list[Achievement]:
    result = []
    for item in raw_list:
        try:
            result.append(Achievement(
                title        = item.get("title", "Untitled"),
                type         = _safe_enum(AchievementType, item.get("type"), AchievementType.OTHER),
                issuer       = item.get("issuer"),
                date_issued  = item.get("date_issued"),
                date_expiry  = item.get("date_expiry"),
                description  = item.get("description"),
                confidence   = float(item.get("confidence", 0.0)),
            ))
        except Exception as exc:
            print(f"[scraper] Skipping invalid achievement: {exc}")
    return result


def _build_events(raw_list: list[dict]) -> list[Event]:
    result = []
    for item in raw_list:
        try:
            result.append(Event(
                title          = item.get("title", "Untitled"),
                type           = _safe_enum(EventType,  item.get("type"),  EventType.OTHER),
                role           = _safe_enum(EventRole,  item.get("role"),  EventRole.ATTENDEE),
                location       = item.get("location"),
                date           = item.get("date"),
                description    = item.get("description"),
                confidence     = float(item.get("confidence", 0.0)),
                source_snippet = item.get("source_snippet"),
            ))
        except Exception as exc:
            print(f"[scraper] Skipping invalid event: {exc}")
    return result


def _safe_enum(enum_class, value, default):
    """Parse an enum value safely, falling back to default on invalid input."""
    if value is None:
        return default
    try:
        return enum_class(value.lower())
    except (ValueError, AttributeError):
        return default


# ── Auth / block detection ────────────────────────────────────────────────────

_AUTH_SIGNALS = ["join now", "sign in", "log in to see", "authwall", "be the first to see"]
_NOT_FOUND    = ["page not found", "this page doesn't exist", "profile is not available"]


def _check_for_blocks(text: str, url: str) -> None:
    lower = text.lower()
    for s in _AUTH_SIGNALS:
        if s in lower:
            raise ValueError(
                f"Auth-wall detected on {url!r}. "
                "Run first_login.py — the session may have expired."
            )
    for s in _NOT_FOUND:
        if s in lower:
            raise ValueError(f"Profile not found or not public: {url!r}")
