"""
faculty_schema.py
==================
Pydantic models for the full faculty profile — existing fields plus the
four new extraction categories: publications, projects, achievements, events.

Design rules reflected here:
  - Publications  ← profile section (papers + patents, typed)
  - Projects      ← profile section
  - Achievements  ← profile section (certifications + awards, typed)
  - Events        ← posts feed only (with source snippet)
  - All items carry a `confidence` float and a `source` label
  - Time filter option C: dated items filtered, year-only always included,
    undated items excluded — enforced in the scraper, not here
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class TimeWindow(int, Enum):
    ONE_MONTH    = 1
    THREE_MONTHS = 3
    SIX_MONTHS   = 6


class PublicationType(str, Enum):
    PAPER   = "paper"
    PATENT  = "patent"
    OTHER   = "other"


class AchievementType(str, Enum):
    CERTIFICATION = "certification"
    AWARD         = "award"
    OTHER         = "other"


class EventType(str, Enum):
    CONFERENCE  = "conference"
    WEBINAR     = "webinar"
    PODCAST     = "podcast"
    PANEL       = "panel"
    WORKSHOP    = "workshop"
    TALK        = "talk"
    OTHER       = "other"


class EventRole(str, Enum):
    SPEAKER   = "speaker"
    PRESENTER = "presenter"
    PANELIST  = "panelist"
    HOST      = "host"
    ATTENDEE  = "attendee"
    OTHER     = "other"


# ── Item models ───────────────────────────────────────────────────────────────

class Publication(BaseModel):
    title:       str                       = Field(...)
    type:        PublicationType           = Field(PublicationType.PAPER)
    publisher:   Optional[str]             = Field(None)
    date:        Optional[str]             = Field(None, description="YYYY-MM or YYYY")
    description: Optional[str]             = Field(None)
    url:         Optional[str]             = Field(None)
    confidence:  float                     = Field(0.0, ge=0.0, le=1.0)
    source:      str                       = Field("profile_section")

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, v):
        return v.strip() if isinstance(v, str) else v


class Project(BaseModel):
    title:         str            = Field(...)
    date_start:    Optional[str]  = Field(None, description="YYYY-MM or YYYY")
    date_end:      Optional[str]  = Field(None, description="YYYY-MM, YYYY, or 'Present'")
    description:   Optional[str]  = Field(None)
    collaborators: list[str]      = Field(default_factory=list)
    url:           Optional[str]  = Field(None)
    confidence:    float          = Field(0.0, ge=0.0, le=1.0)
    source:        str            = Field("profile_section")

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, v):
        return v.strip() if isinstance(v, str) else v


class Achievement(BaseModel):
    title:        str              = Field(...)
    type:         AchievementType  = Field(AchievementType.OTHER)
    issuer:       Optional[str]    = Field(None)
    date_issued:  Optional[str]    = Field(None, description="YYYY-MM or YYYY")
    date_expiry:  Optional[str]    = Field(None)
    description:  Optional[str]    = Field(None)
    confidence:   float            = Field(0.0, ge=0.0, le=1.0)
    source:       str              = Field("profile_section")

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, v):
        return v.strip() if isinstance(v, str) else v


class Event(BaseModel):
    title:          str             = Field(...)
    type:           EventType       = Field(EventType.OTHER)
    role:           EventRole       = Field(EventRole.ATTENDEE)
    location:       Optional[str]   = Field(None)
    date:           Optional[str]   = Field(None, description="YYYY-MM or YYYY")
    description:    Optional[str]   = Field(None)
    confidence:     float           = Field(0.0, ge=0.0, le=1.0)
    source:         str             = Field("post")
    source_snippet: Optional[str]   = Field(None, max_length=300)

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, v):
        return v.strip() if isinstance(v, str) else v


# ── Master profile model ──────────────────────────────────────────────────────

class FacultyProfile(BaseModel):
    # Core identity
    name:        Optional[str]  = Field(None)
    title:       Optional[str]  = Field(None)
    affiliation: Optional[str]  = Field(None)
    location:    Optional[str]  = Field(None)
    summary:     Optional[str]  = Field(None, max_length=500)
    confidence:  float          = Field(0.0, ge=0.0, le=1.0)

    # New category lists
    publications: list[Publication] = Field(default_factory=list)
    projects:     list[Project]     = Field(default_factory=list)
    achievements: list[Achievement] = Field(default_factory=list)
    events:       list[Event]       = Field(default_factory=list)

    # Metadata
    time_window_months: Optional[int] = Field(None)
    error:              Optional[str] = Field(None)

    @field_validator("name", "title", "affiliation", "location", mode="before")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("summary", mode="before")
    @classmethod
    def _truncate_summary(cls, v):
        if isinstance(v, str):
            return v.strip()[:500] or None
        return v

    @model_validator(mode="after")
    def _warn_if_empty(self) -> "FacultyProfile":
        if not self.name and not self.title:
            print("[schema] WARNING: Name and title are both null.")
        return self

    @property
    def is_complete(self) -> bool:
        return bool(self.name and self.title and self.affiliation)

    @property
    def total_items(self) -> int:
        return len(self.publications) + len(self.projects) + len(self.achievements) + len(self.events)

    def display(self) -> str:
        parts = [self.name or "Unknown"]
        if self.title:       parts.append(self.title)
        if self.affiliation: parts.append(f"@ {self.affiliation}")
        return " | ".join(parts)


# ── API models ────────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    profile: str = Field(
        ...,
        description="LinkedIn slug or full URL. Must be your own profile or a 1st-degree connection.",
        examples=["williamhgates"],
    )
    time_window: TimeWindow = Field(
        default=TimeWindow.THREE_MONTHS,
        description="1, 3, or 6 months. Year-only dated items always included regardless.",
    )


class ScrapeResponse(BaseModel):
    success:     bool
    profile_url: str
    data:        Optional[FacultyProfile] = None
    error:       Optional[str]            = None
