"""
prompts.py
===========
All LLM prompts in one place — single source of truth.

Two prompts:
  1. PROFILE SECTIONS  → extracts publications, projects, achievements
                         from the full profile page visible text.

  2. EVENTS            → extracts public appearances from a batch of
                         post texts collected from the activity feed.

Keeping prompts separate from extractor logic means you can tune them
without touching any scraping or validation code.
"""

from __future__ import annotations


# ── Prompt 1: Profile sections ────────────────────────────────────────────────

_PROFILE_SYSTEM = """\
You are a precise academic profile data extractor.
You receive VISIBLE TEXT scraped from a LinkedIn profile page.
Your ONLY job is to extract publications, projects, and achievements.

STRICT RULES:
1. Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
2. Use JSON null for any field you cannot find. NEVER invent data.
3. Copy titles and names exactly as written — do not paraphrase.
4. For dates: use "YYYY-MM" if month is available, "YYYY" if only year is present, null if absent.
5. For the `type` field of publications: "paper" for research articles, "patent" for patents, "other" for everything else.
6. For the `type` field of achievements: "certification" for certifications/courses, "award" for honors/awards/prizes, "other" for everything else.
7. If a section does not exist on the page return an empty array [] for that key.
8. Patents found in a Patents section belong in publications[] with type "patent".
9. Both the Certifications section AND the Honors & Awards section feed into achievements[].
"""

_PROFILE_USER = """\
Extract structured academic data from the following LinkedIn profile text.

Return a JSON object with exactly these three keys:

{{
  "publications": [
    {{
      "title": "string — exact title",
      "type": "paper | patent | other",
      "publisher": "journal / conference / patent office or null",
      "date": "YYYY-MM or YYYY or null",
      "description": "short abstract or description or null",
      "url": "absolute URL or null",
      "confidence": 0.0–1.0
    }}
  ],
  "projects": [
    {{
      "title": "string — exact project name",
      "date_start": "YYYY-MM or YYYY or null",
      "date_end": "YYYY-MM or YYYY or 'Present' or null",
      "description": "project description or null",
      "collaborators": ["name1", "name2"],
      "url": "absolute URL or null",
      "confidence": 0.0–1.0
    }}
  ],
  "achievements": [
    {{
      "title": "string — exact certification or award name",
      "type": "certification | award | other",
      "issuer": "issuing organisation or null",
      "date_issued": "YYYY-MM or YYYY or null",
      "date_expiry": "YYYY-MM or YYYY or null",
      "description": "additional context or null",
      "confidence": 0.0–1.0
    }}
  ]
}}

Profile text (may be truncated):
---
{text}
---
"""


def get_profile_sections_prompt(visible_text: str) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for profile section extraction.
    Text is passed as-is — truncation happens in the extractor.
    """
    return _PROFILE_SYSTEM, _PROFILE_USER.format(text=visible_text)


# ── Prompt 2: Events from posts ───────────────────────────────────────────────

_EVENTS_SYSTEM = """\
You are a precise event extraction assistant.
You receive a list of LinkedIn posts as JSON, each with "text" and "date" fields.
Your job is to identify posts that describe a PUBLIC APPEARANCE by the profile owner.

A public appearance includes: conferences, seminars, webinars, workshops, podcast appearances,
panel discussions, invited talks, keynote speeches, award ceremonies attended.

It does NOT include: sharing someone else's article, general opinions,
job announcements, congratulating others, generic motivational posts.

STRICT RULES:
1. Return ONLY a valid JSON object — no markdown, no explanation, no code fences.
2. Only extract events where you are confident the profile owner personally attended or participated.
3. Use the post's date field as the event date when no other date is mentioned.
4. source_snippet must be the most relevant excerpt (max 250 chars) from the post that proves this is an event.
5. If no posts describe a public appearance, return {"events": []}.
6. Do not invent event names — use the name as stated in the post, or a clear descriptive title.
"""

_EVENTS_USER = """\
Identify public appearances from the following LinkedIn posts.

Return a JSON object with exactly this structure:

{{
  "events": [
    {{
      "title": "exact event name or clear descriptive title",
      "type": "conference | webinar | podcast | panel | workshop | talk | other",
      "role": "speaker | presenter | panelist | host | attendee | other",
      "location": "city name or 'Online' or null",
      "date": "YYYY-MM or YYYY inferred from context or post date",
      "description": "what the person presented or discussed, or null",
      "confidence": 0.0–1.0,
      "source_snippet": "most relevant excerpt from the post (max 250 chars)"
    }}
  ]
}}

Posts to analyse:
---
{posts_json}
---
"""


def get_events_prompt(posts: list[dict]) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for event extraction from posts.

    posts: list of {"text": str, "date": str | None} dicts
           already filtered to the requested time window.
    """
    import json
    posts_json = json.dumps(posts, indent=2, ensure_ascii=False)
    return _EVENTS_SYSTEM, _EVENTS_USER.format(posts_json=posts_json)
