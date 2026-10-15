"""
scholar_schema.py
==================
Pydantic models for a Google Scholar profile.

Completely independent from faculty_schema.py — Scholar data is exposed
via its own endpoint and never merged with LinkedIn data automatically.

Field coverage (everything Scholar's profile page exposes):
  - Identity       : name, affiliation, verified email, research interests
  - Metrics        : citations, h-index, i10-index  (all-time + since 2019)
  - Citation graph : year-by-year citation counts  (bar chart data)
  - Publications   : capped at 20, with per-paper citation count + URL
  - Co-authors     : sidebar author cards
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ScholarMetrics(BaseModel):
    """
    The metrics table rendered on every Scholar profile.
    Scholar shows two columns: all-time and since 2019.
    Both are captured — they tell different stories
    (total career impact vs recent activity).
    """
    citations_total:      Optional[int] = Field(None, description="Total citations (all time)")
    citations_since_2019: Optional[int] = Field(None, description="Citations since 2019")
    h_index_total:        Optional[int] = Field(None, description="h-index (all time)")
    h_index_since_2019:   Optional[int] = Field(None, description="h-index since 2019")
    i10_index_total:      Optional[int] = Field(None, description="i10-index (all time)")
    i10_index_since_2019: Optional[int] = Field(None, description="i10-index since 2019")


class CitationYear(BaseModel):
    """One bar in the citation-per-year chart embedded on the Scholar profile."""
    year:      int = Field(..., description="Calendar year")
    citations: int = Field(..., description="Citations received that year")


class ScholarPublication(BaseModel):
    """
    One row from the Scholar profile publications table.
    Capped at 20 per project design decision.
    scholar_url → Scholar paper detail page (full author list, abstract, citers).
    """
    title:       str           = Field(...,  description="Publication title")
    authors:     Optional[str] = Field(None, description="Author string as shown by Scholar")
    venue:       Optional[str] = Field(None, description="Journal or conference name")
    year:        Optional[int] = Field(None, description="Publication year")
    citations:   Optional[int] = Field(None, description="Times cited")
    scholar_url: Optional[str] = Field(None, description="URL to Scholar paper detail page")

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("citations", mode="before")
    @classmethod
    def coerce_citations(cls, v):
        """
        Scholar quirks:
          '*'  → <10 citations, exact count hidden → return None
          ''   → no citations row shown → return None
        """
        if v is None:
            return None
        if isinstance(v, int):
            return v
        s = str(v).strip()
        return int(s) if s.isdigit() else None


class CoAuthor(BaseModel):
    """One entry from the co-authors sidebar (Scholar shows up to ~20)."""
    name:        str           = Field(...,  description="Co-author full name")
    affiliation: Optional[str] = Field(None, description="Co-author's listed affiliation")
    scholar_url: Optional[str] = Field(None, description="Link to co-author's Scholar profile")

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v):
        return v.strip() if isinstance(v, str) else v


class ScholarProfile(BaseModel):
    """
    Complete structured representation of a Google Scholar profile.
    All list fields default to empty — valid profile can have no co-authors etc.
    """
    # Identity
    name:               Optional[str]         = Field(None)
    affiliation:        Optional[str]         = Field(None)
    verified_email:     Optional[str]         = Field(None, description="e.g. 'Verified email at mit.edu'")
    research_interests: list[str]             = Field(default_factory=list)
    scholar_url:        Optional[str]         = Field(None, description="Canonical URL of this profile")

    # Metrics
    metrics:            Optional[ScholarMetrics]      = Field(None)

    # Citation graph
    citation_graph:     list[CitationYear]            = Field(default_factory=list)

    # Publications (capped at 20)
    publications:       list[ScholarPublication]      = Field(default_factory=list)

    # Co-authors
    co_authors:         list[CoAuthor]                = Field(default_factory=list)

    # Metadata
    scrape_error:       Optional[str]                 = Field(None)

    @field_validator("name", "affiliation", "verified_email", mode="before")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @model_validator(mode="after")
    def _warn_if_empty(self) -> "ScholarProfile":
        if not self.name and not self.publications:
            print(
                "[schema] WARNING: Scholar profile has no name and no publications — "
                "possible CAPTCHA block or invalid URL."
            )
        return self

    @property
    def is_complete(self) -> bool:
        return bool(self.name and self.metrics and self.publications)

    @property
    def total_publications(self) -> int:
        return len(self.publications)

    def display(self) -> str:
        parts = [self.name or "Unknown Scholar"]
        if self.affiliation:
            parts.append(self.affiliation)
        if self.metrics and self.metrics.citations_total is not None:
            parts.append(f"Citations: {self.metrics.citations_total:,}")
        if self.metrics and self.metrics.h_index_total is not None:
            parts.append(f"h-index: {self.metrics.h_index_total}")
        return " | ".join(parts)


# ── API models ────────────────────────────────────────────────────────────────

class ScholarScrapeRequest(BaseModel):
    url: str = Field(
        ...,
        description="Full Google Scholar profile URL.",
        examples=["https://scholar.google.com/citations?user=ABC123&hl=en"],
    )


class ScholarScrapeResponse(BaseModel):
    success:     bool
    profile_url: str
    data:        Optional[ScholarProfile] = None
    error:       Optional[str]            = None
