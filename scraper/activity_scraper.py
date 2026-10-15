"""
activity_scraper.py
====================
Scrapes the LinkedIn activity feed for a profile and returns a filtered
list of post texts + parsed dates, ready to be sent to the LLM for
event extraction.

URL pattern: https://www.linkedin.com/in/{slug}/recent-activity/all/

Key design decisions:
  - Uses data-urn attribute selectors to find post cards — URNs are part
    of LinkedIn's core data model and far more stable than class names.
  - Reads aria-label attributes on timestamp spans — accessibility labels
    are stable because they must satisfy screen-reader requirements.
  - Bounded scroll budget (MAX_SCROLLS) prevents runaway scraping on
    very active profiles while still covering the requested time window.
  - Stops early when the oldest visible post is older than the cutoff,
    so sparse posters don't waste the full scroll budget.
  - Deduplicates posts by a text fingerprint to avoid counting the same
    card twice across scroll iterations.
  - Gracefully returns [] if the feed is private or the profile is not
    a connection — callers must handle empty list without erroring.
"""

from __future__ import annotations

import asyncio
import random
import re
from datetime import datetime, timedelta
from typing import Optional

from dateutil.relativedelta import relativedelta
from patchright.async_api import Page

LINKEDIN_BASE = "https://www.linkedin.com"

# Maximum number of scroll iterations regardless of date cutoff.
# Each scroll reveals roughly 3-6 posts. 25 scrolls ≈ 75-150 posts.
MAX_SCROLLS = 25

# Minimum chars a post must have to be worth sending to the LLM.
MIN_POST_CHARS = 30


# ── ActivityScraper ───────────────────────────────────────────────────────────

class ActivityScraper:
    """
    Navigates to the LinkedIn activity feed, scrolls through posts within
    the requested time window, and returns structured post data.

    Usage:
        scraper = ActivityScraper()
        posts = await scraper.scrape(page, "williamhgates", months=3)
        # posts = [{"text": "...", "date": "2024-11", "raw_date": "2 months ago"}, ...]
    """

    async def scrape(
        self,
        page:    Page,
        slug:    str,
        months:  int,
    ) -> list[dict]:
        """
        Navigate to the activity feed and return posts within the time window.

        Args:
            page:   An open Patchright page (already authenticated).
            slug:   LinkedIn username slug (e.g. "williamhgates").
            months: How far back to look (1, 3, or 6).

        Returns:
            List of {"text": str, "date": str | None, "raw_date": str} dicts.
            Returns [] if feed is inaccessible or no posts found.
        """
        cutoff   = datetime.now() - relativedelta(months=months)
        feed_url = f"{LINKEDIN_BASE}/in/{slug}/recent-activity/all/"

        print(f"[activity] Navigating to feed: {feed_url}")
        print(f"[activity] Cutoff date: {cutoff.strftime('%Y-%m-%d')} ({months} months)")

        try:
            await page.goto(feed_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(random.uniform(2.0, 3.0))
        except Exception as exc:
            print(f"[activity] Navigation failed: {exc}")
            return []

        # Check we actually landed on the activity feed
        if not await self._is_feed_accessible(page, slug):
            return []

        posts = await self._collect_posts(page, cutoff)
        print(f"[activity] Collected {len(posts)} posts within the time window.")
        return posts

    # ── Feed accessibility check ──────────────────────────────────────────────

    async def _is_feed_accessible(self, page: Page, slug: str) -> bool:
        """
        Returns False if LinkedIn redirected away from the activity feed —
        which happens when the profile is private or not a connection.
        """
        current_url = page.url

        # Redirected to auth wall
        if "authwall" in current_url or "login" in current_url:
            print("[activity] Redirected to auth wall — session may have expired.")
            return False

        # Redirected away from the activity feed URL
        if f"/in/{slug}" not in current_url and "recent-activity" not in current_url:
            print(f"[activity] Unexpected redirect to: {current_url}")
            return False

        # Check for explicit "no posts" signal in page text
        try:
            body_text = await page.inner_text("body")
            no_post_signals = [
                "no posts yet",
                "no activity to show",
                "this member's activity is only visible",
            ]
            for signal in no_post_signals:
                if signal.lower() in body_text.lower():
                    print(f"[activity] Feed not visible: '{signal}' detected.")
                    return False
        except Exception:
            pass

        return True

    # ── Post collection loop ──────────────────────────────────────────────────

    async def _collect_posts(self, page: Page, cutoff: datetime) -> list[dict]:
        """
        Scroll through the feed, collecting post text and timestamps.
        Stops when either:
          a) The oldest visible post is older than `cutoff`, OR
          b) MAX_SCROLLS iterations are reached.

        Deduplicates by a 100-char text fingerprint so repeated cards
        across scroll positions don't appear twice.
        """
        collected:   list[dict] = []
        seen_keys:   set[str]   = set()
        past_cutoff: bool       = False

        for scroll_num in range(MAX_SCROLLS):
            # Extract all currently visible post cards
            cards = await self._extract_post_cards(page)

            for card in cards:
                key = card["text"][:100]
                if key in seen_keys or len(card["text"]) < MIN_POST_CHARS:
                    continue
                seen_keys.add(key)

                # Parse the timestamp
                parsed_dt = parse_post_timestamp(card["raw_date"])
                card["date"] = _dt_to_str(parsed_dt)

                # Stop scrolling if we've gone past the window
                if parsed_dt and parsed_dt < cutoff:
                    past_cutoff = True
                    break

                # Only keep posts within the window (or undatable ones)
                if parsed_dt is None or parsed_dt >= cutoff:
                    collected.append(card)

            if past_cutoff:
                print(f"[activity] Reached cutoff at scroll {scroll_num + 1}.")
                break

            # Scroll down one viewport and wait for lazy load
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(random.uniform(1.5, 2.5))

            # Check if we've hit the bottom of the page (no more content)
            new_height = await page.evaluate("document.body.scrollHeight")
            if scroll_num > 0 and not cards:
                print("[activity] No new cards found — end of feed reached.")
                break

        return collected

    # ── Post card extraction ──────────────────────────────────────────────────

    async def _extract_post_cards(self, page: Page) -> list[dict]:
        """
        Find post card elements on the page and extract their text + timestamp.

        Strategy:
          1. Locate elements with data-urn containing "activity" — these are
             the post card root elements LinkedIn consistently uses.
          2. For each card, read inner_text() for the post body.
          3. Find the timestamp via aria-label on span/time elements inside
             the card — aria-labels are stable across LinkedIn redesigns.
        """
        cards = []

        try:
            # data-urn="urn:li:activity:..." is LinkedIn's stable post identifier
            card_locators = page.locator('[data-urn*="activity"]')
            count = await card_locators.count()

            for i in range(count):
                card = card_locators.nth(i)
                try:
                    text      = await card.inner_text(timeout=3000)
                    raw_date  = await self._extract_timestamp(card)

                    # Clean up the text
                    text = _clean_post_text(text)

                    if text:
                        cards.append({
                            "text":     text,
                            "raw_date": raw_date or "",
                            "date":     None,  # filled in by caller after parsing
                        })
                except Exception:
                    continue

        except Exception as exc:
            print(f"[activity] Card extraction error: {exc}")

        return cards

    async def _extract_timestamp(self, card) -> Optional[str]:
        """
        Extract the timestamp string from a post card.

        LinkedIn renders timestamps in span elements with aria-label attributes
        like aria-label="2 months ago" or aria-label="June 12, 2024".
        The abbreviated visible text ("2mo", "Jun 12") is less useful than
        the full aria-label, so we prefer aria-label.
        """
        try:
            # Try aria-label spans first — most complete date info
            time_spans = card.locator("span[aria-label]")
            span_count = await time_spans.count()

            for i in range(span_count):
                span = time_spans.nth(i)
                label = await span.get_attribute("aria-label", timeout=1000) or ""
                text  = await span.inner_text(timeout=1000)

                # A timestamp span has short visible text ("2mo", "1w", "Jun 12")
                # and a longer aria-label. If visible text looks like a timestamp,
                # prefer the full aria-label for parsing accuracy.
                if label and _looks_like_timestamp(text.strip()):
                    return label.strip()

            # Fall back: look for a <time> element
            time_el = card.locator("time")
            if await time_el.count() > 0:
                dt_attr = await time_el.first.get_attribute("datetime", timeout=1000)
                if dt_attr:
                    return dt_attr

        except Exception:
            pass

        return None


# ── Timestamp parsing ─────────────────────────────────────────────────────────

_MONTHS_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_post_timestamp(raw: str) -> Optional[datetime]:
    """
    Convert a LinkedIn timestamp string to a Python datetime.

    Handles all formats LinkedIn uses:
      Abbreviated : "2h", "3d", "1w", "2mo", "1yr"
      Full aria   : "2 months ago", "3 days ago", "June 12, 2024"
      ISO         : "2024-06-12T10:30:00" (from datetime attributes)
      Partial     : "June 12" (assumes current year)
      Year-only   : "2024" → treated as Jan 1 of that year
    """
    if not raw:
        return None

    s   = raw.strip()
    now = datetime.now()

    # ── ISO datetime from <time datetime="..."> ────────────────────────────
    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if iso_match:
        try:
            return datetime(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        except ValueError:
            pass

    sl = s.lower()

    # ── Abbreviated relative: "2h", "3d", "1w", "2mo", "1yr" ─────────────
    abbrev = re.match(r"^(\d+)(h|d|w|mo|yr)$", sl)
    if abbrev:
        n, unit = int(abbrev.group(1)), abbrev.group(2)
        if unit == "h":  return now - timedelta(hours=n)
        if unit == "d":  return now - timedelta(days=n)
        if unit == "w":  return now - timedelta(weeks=n)
        if unit == "mo": return now - relativedelta(months=n)
        if unit == "yr": return now - relativedelta(years=n)

    # ── Full relative: "2 months ago", "3 days ago", "just now" ──────────
    if sl in ("just now", "now", "moments ago"):
        return now

    full_rel = re.match(
        r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", sl
    )
    if full_rel:
        n, unit = int(full_rel.group(1)), full_rel.group(2)
        if unit in ("second", "minute"): return now
        if unit == "hour":  return now - timedelta(hours=n)
        if unit == "day":   return now - timedelta(days=n)
        if unit == "week":  return now - timedelta(weeks=n)
        if unit == "month": return now - relativedelta(months=n)
        if unit == "year":  return now - relativedelta(years=n)

    # ── "June 12, 2024" or "Jun 12, 2024" ────────────────────────────────
    full_date = re.match(
        r"([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", sl
    )
    if full_date:
        mon_name, day, year = full_date.group(1)[:3], int(full_date.group(2)), int(full_date.group(3))
        mon = _MONTHS_MAP.get(mon_name)
        if mon:
            try:
                return datetime(year, mon, day)
            except ValueError:
                pass

    # ── "June 12" or "Jun 12" (assume current year, roll back if future) ──
    partial = re.match(r"([a-z]+)\s+(\d{1,2})$", sl)
    if partial:
        mon_name, day = partial.group(1)[:3], int(partial.group(2))
        mon = _MONTHS_MAP.get(mon_name)
        if mon:
            try:
                dt = now.replace(month=mon, day=day, hour=0, minute=0, second=0, microsecond=0)
                if dt > now:
                    dt = dt.replace(year=now.year - 1)
                return dt
            except ValueError:
                pass

    # ── Year-only: "2024" → Jan 1 of that year ────────────────────────────
    year_only = re.match(r"^(\d{4})$", sl)
    if year_only:
        try:
            return datetime(int(year_only.group(1)), 1, 1)
        except ValueError:
            pass

    return None


# ── Utilities ─────────────────────────────────────────────────────────────────

def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to "YYYY-MM" string, or None."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m")


def _looks_like_timestamp(text: str) -> bool:
    """
    Returns True if the visible text of a span looks like a LinkedIn timestamp.
    Used to decide whether to prefer the aria-label over the visible text.
    """
    # Match abbreviated formats: "2h", "3d", "1w", "2mo", "1yr"
    if re.match(r"^\d+(h|d|w|mo|yr)$", text.lower()):
        return True
    # Match short date formats: "Jun 12", "Jan 5"
    if re.match(r"^[A-Za-z]{3}\s+\d{1,2}$", text):
        return True
    return False


def _clean_post_text(raw: str) -> str:
    """
    Clean inner_text() output from a LinkedIn post card.
    Removes UI chrome (reaction counts, "Like", "Comment" buttons, etc.)
    that inner_text() picks up alongside the actual post content.
    """
    # Split into lines, remove UI-only lines
    ui_patterns = [
        r"^\d+$",                           # reaction/comment counts
        r"^(like|comment|repost|send)$",    # button labels
        r"^\d+\s+(comment|reaction|repost)", # "14 comments"
        r"^(see more|show more|…)$",        # expand links
        r"^follow$",                         # follow button
        r"^\.\.\.$",
    ]
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        skip = False
        for pattern in ui_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                skip = True
                break
        if not skip:
            lines.append(stripped)

    return " ".join(lines).strip()
