"""Pydantic models used across the public API and persistence layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MemoryKind = Literal["short_term", "long_term", "episodic", "semantic", "summary"]


class MemoryRecord(BaseModel):
    """A persisted memory plus retrieval metadata.

    ``embedding`` is included for import/export fidelity, but searches omit it
    by default to keep result objects small.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    namespace: str = "default"
    text: str
    embedding: list[float] | None = Field(default=None, repr=False)
    metadata: dict[str, Any] | None = None
    memory_type: MemoryKind = "long_term"
    source_ids: list[str] = Field(default_factory=list)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime
    accessed_at: datetime
    access_count: int = Field(default=0, ge=0)
    ttl_days: int | None = Field(default=None, ge=1)
    score: float | None = Field(default=None, ge=0.0)
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("memory text must not be empty")
        return text

    @field_validator("namespace")
    @classmethod
    def namespace_must_not_be_empty(cls, value: str) -> str:
        namespace = value.strip()
        if not namespace:
            raise ValueError("namespace must not be empty")
        return namespace

    @field_validator("source_ids")
    @classmethod
    def source_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class SaveRequest(BaseModel):
    """Validated input for saving a memory."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] | None = None
    memory_type: MemoryKind = "long_term"
    source_ids: list[str] = Field(default_factory=list)
    ttl_days: int | None = Field(default=None, ge=1, le=365_000)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    namespace: str = "default"


class SearchRequest(BaseModel):
    """Validated input for searching memories."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=20_000)
    k: int = Field(default=5, ge=1, le=100)
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    filters: dict[str, Any] | None = None
    namespace: str = "default"
    memory_types: list[MemoryKind] | None = None


class SummaryResult(BaseModel):
    """Result returned by memory compaction/summarization."""

    model_config = ConfigDict(extra="forbid")

    summary_id: str | None
    source_ids: list[str]
    text: str | None
    deleted_sources: int = 0


class MemoryStats(BaseModel):
    """Storage and namespace statistics."""

    model_config = ConfigDict(extra="forbid")

    path: str
    namespace: str | None = None
    count: int
    deleted_count: int
    size_mb: float
    vector_index: str
    dimension: int


def datetime_from_timestamp(value: int | float) -> datetime:
    """Convert a Unix timestamp into a timezone-aware UTC datetime."""

    return datetime.fromtimestamp(value, tz=timezone.utc)


def timestamp_from_datetime(value: datetime) -> int:
    """Convert a datetime to an integer Unix timestamp."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())
