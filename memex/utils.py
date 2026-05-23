"""Small internal utilities kept dependency-free."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from memex.errors import MemexValidationError

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def utc_timestamp() -> int:
    """Return the current Unix timestamp in seconds."""

    return int(time.time())


def default_data_dir() -> Path:
    """Return the default memex data directory without importing appdirs."""

    override = os.environ.get("MEMEX_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".memex"


def default_db_path() -> Path:
    """Return the default SQLite database path."""

    override = os.environ.get("MEMEX_DB")
    if override:
        return Path(override).expanduser()
    return default_data_dir() / "memex.db"


def normalize_namespace(namespace: str | None) -> str:
    """Validate and normalize a namespace."""

    value = (namespace or "default").strip()
    if not _NAMESPACE_RE.match(value):
        raise MemexValidationError(
            "namespace must be 1-128 characters and contain only letters, "
            "numbers, underscore, dot, colon, or dash"
        )
    return value


def normalize_text(text: str, *, max_chars: int = 100_000) -> str:
    """Validate and normalize user-provided text."""

    if not isinstance(text, str):
        raise MemexValidationError("text must be a string")
    value = text.strip()
    if not value:
        raise MemexValidationError("text must not be empty")
    if len(value) > max_chars:
        raise MemexValidationError(f"text exceeds the {max_chars} character limit")
    return value


def validate_metadata(
    metadata: dict[str, Any] | None, *, max_bytes: int = 65_536
) -> dict[str, Any] | None:
    """Ensure metadata is JSON-serializable and bounded."""

    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise MemexValidationError("metadata must be a JSON object")
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > max_bytes:
        raise MemexValidationError(f"metadata exceeds the {max_bytes} byte limit")
    json.loads(encoded)
    return metadata


def metadata_to_json(metadata: dict[str, Any] | None) -> str | None:
    """Serialize metadata with stable key ordering."""

    if metadata is None:
        return None
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def metadata_from_json(value: str | None) -> dict[str, Any] | None:
    """Safely parse metadata stored in SQLite."""

    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise MemexValidationError("stored metadata is not a JSON object")
    return parsed


def secure_parent_dir(path: Path) -> Path:
    """Create the database parent directory and apply private permissions."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.name != "nt":
            resolved.parent.chmod(0o700)
    except OSError:
        pass
    return resolved


def secure_file(path: Path) -> None:
    """Best-effort user-only permissions for the SQLite database."""

    try:
        if path.exists() and os.name != "nt":
            path.chmod(0o600)
    except OSError:
        pass


def validate_embedding(vector: Iterable[float], *, dimension: int | None = None) -> list[float]:
    """Return a finite float list and enforce dimensionality when provided."""

    values = [float(x) for x in vector]
    if dimension is not None and len(values) != dimension:
        raise MemexValidationError(
            f"embedding dimension mismatch: expected {dimension}, got {len(values)}"
        )
    if not values:
        raise MemexValidationError("embedding must not be empty")
    if not all(math.isfinite(x) for x in values):
        raise MemexValidationError("embedding contains NaN or infinite values")
    return values


def pack_embedding(vector: Iterable[float], *, dimension: int | None = None) -> bytes:
    """Pack a float vector into little-endian float32 bytes."""

    values = validate_embedding(vector, dimension=dimension)
    packed = array("f", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def unpack_embedding(blob: bytes, *, dimension: int | None = None) -> list[float]:
    """Unpack little-endian float32 bytes into a Python list."""

    if len(blob) % 4 != 0:
        raise MemexValidationError("embedding blob length is not divisible by 4")
    values = array("f")
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    result = [float(x) for x in values]
    if dimension is not None and len(result) != dimension:
        raise MemexValidationError(
            f"embedding dimension mismatch: expected {dimension}, got {len(result)}"
        )
    return result


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two vectors."""

    if len(left) != len(right):
        raise MemexValidationError(
            f"cannot compare vectors with dimensions {len(left)} and {len(right)}"
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def normalized(values: list[float]) -> list[float]:
    """Return an L2-normalized copy of ``values``."""

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def tokenize(text: str) -> list[str]:
    """Tokenize text for the deterministic fallback embedder."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def metadata_matches(metadata: dict[str, Any] | None, filters: dict[str, Any] | None) -> bool:
    """Return whether metadata satisfies all exact-match filters."""

    if not filters:
        return True
    if not metadata:
        return False
    return all(metadata.get(key) == value for key, value in filters.items())


def safe_json_loads(raw: str) -> Any:
    """Parse JSON and normalize JSONDecodeError into ValueError."""

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemexValidationError(f"invalid JSON: {exc.msg}") from exc
