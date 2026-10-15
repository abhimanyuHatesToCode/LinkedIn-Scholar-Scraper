"""
ai_extractor.py
================
Two focused LLM calls via OpenRouter (OpenAI-compatible):
  1. extract_profile_sections() → publications, projects, achievements
  2. extract_events()           → events from posts feed

One client instance is shared across both calls — the httpx connection
pool is reused, keeping latency down on sequential scrapes.

Model: configured via OPENROUTER_MODEL in .env.
       gpt-4o-mini is the default — cheap and accurate for structured extraction.
       Swap to gpt-4o or claude-3-5-sonnet for higher accuracy on complex profiles.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

from extraction.prompts import get_profile_sections_prompt, get_events_prompt
from utils.file_utils import save_data

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Profile text is trimmed to this before sending — controls token cost.
# 14 000 chars ≈ ~3 500 tokens, well within gpt-4o-mini's 128k context.
MAX_PROFILE_CHARS = int(os.getenv("MAX_TEXT_CHARS", "14000"))

# Posts are sent as a JSON array — cap total chars to avoid context overflow.
MAX_POSTS_CHARS = int(os.getenv("MAX_POSTS_CHARS", "10000"))


# ── AIExtractor ───────────────────────────────────────────────────────────────

class AIExtractor:
    """
    Stateless wrapper around two OpenRouter LLM calls.
    Instantiate once at server startup; reuse across all requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model:   Optional[str] = None,
    ) -> None:
        key = api_key or OPENROUTER_API_KEY
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or pass api_key= explicitly."
            )

        self._model  = model or OPENROUTER_MODEL
        self._client = AsyncOpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/local/linkedin-faculty-scraper",
                "X-Title":      "LinkedIn Faculty Scraper",
            },
        )
        print(f"[extractor] Using model: {self._model}")

    # ── Public API ────────────────────────────────────────────────────────────

    async def extract_profile_sections(self, visible_text: str) -> dict:
        """
        Extract publications, projects, and achievements from profile page text.

        Returns a dict with keys: publications, projects, achievements.
        Each value is a list of dicts matching the corresponding Pydantic model.
        Returns empty lists (not errors) when a section is absent.
        """
        trimmed   = visible_text[:MAX_PROFILE_CHARS]
        system, user = get_profile_sections_prompt(trimmed)

        print(f"[extractor] Profile sections call — {len(trimmed):,} chars of text")
        raw = await self._llm_call(system, user, max_tokens=2048)
        result = _parse_json(raw, context="profile_sections")

        # Guarantee the three keys always exist even if LLM omits them
        return {
            "publications": result.get("publications") or [],
            "projects":     result.get("projects")     or [],
            "achievements": result.get("achievements") or [],
        }

    async def extract_events(self, posts: list[dict]) -> list[dict]:
        """
        Extract public appearance events from a list of post dicts.

        posts: [{"text": str, "date": str | None}, ...]
               already filtered to the time window by the activity scraper.

        Returns a list of event dicts matching the Event Pydantic model.
        Returns [] when no events are found or posts list is empty.
        """
        if not posts:
            print("[extractor] No posts to analyse for events — skipping LLM call.")
            return []

        # Serialise posts to JSON string and trim to budget
        posts_text = json.dumps(posts, ensure_ascii=False)
        if len(posts_text) > MAX_POSTS_CHARS:
            # Trim by dropping posts from the end (oldest first since list is
            # chronological desc) until we're under budget
            trimmed_posts = posts[:]
            while trimmed_posts and len(json.dumps(trimmed_posts, ensure_ascii=False)) > MAX_POSTS_CHARS:
                trimmed_posts.pop()
            posts = trimmed_posts
            print(f"[extractor] Posts trimmed to {len(posts)} items to fit context budget.")

        system, user = get_events_prompt(posts)
        print(f"[extractor] Events call — {len(posts)} posts")
        raw    = await self._llm_call(system, user, max_tokens=2048)
        result = _parse_json(raw, context="events")

        save_data(result)

        return result.get("events") or []

    async def extract_identity(self, visible_text: str) -> dict:
        """
        Extract core identity fields: name, title, affiliation, location, summary.
        This is a lightweight call — same as the original v1 extractor behaviour.
        Kept separate so it can be called independently if needed.
        """
        system = (
            "You are a precise data extraction assistant. "
            "Return ONLY a valid JSON object — no markdown, no explanation. "
            "Use null for missing fields. Never invent data."
        )
        user = f"""\
Extract the following fields from this LinkedIn profile text:
  name        - full name
  title       - current job title or academic position
  affiliation - current employer or university
  location    - city or region
  summary     - first 300 chars of the About section or null
  confidence  - float 0.0–1.0 reflecting completeness

Return only JSON with exactly those six keys.

Profile text:
---
{visible_text[:MAX_PROFILE_CHARS]}
---
"""
        raw = await self._llm_call(system, user, max_tokens=512)
        return _parse_json(raw, context="identity")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _llm_call(
        self,
        system:     str,
        user:       str,
        max_tokens: int = 2048,
    ) -> str:
        """Single OpenRouter completion call. Returns raw response string."""
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,   # deterministic — no hallucinated facts
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        content = response.choices[0].message.content or ""
        print(f"[extractor] Response: {len(content)} chars — preview: {content[:120]}…")
        return content


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _parse_json(raw: str, context: str = "") -> dict:
    """
    Safely parse JSON from LLM output, handling two common failure modes:
      1. Model wraps in ```json … ``` fences
      2. Model adds preamble text before the opening brace
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$",         "", cleaned).strip()

    # Find outermost { … }
    start = cleaned.find("{")
    end   = cleaned.rfind("}")

    if start == -1 or end == -1:
        print(f"[extractor] WARNING: No JSON object in response for '{context}'")
        return {}

    try:
        return json.loads(cleaned[start: end + 1])
    except json.JSONDecodeError as exc:
        print(f"[extractor] WARNING: JSON parse error ({exc}) for '{context}'")
        return {}
