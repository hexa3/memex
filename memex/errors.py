"""Exception hierarchy for memex."""

from __future__ import annotations


class MemexError(Exception):
    """Base class for all memex-specific errors."""


class MemexValidationError(MemexError, ValueError):
    """Raised when user input fails validation."""


class StorageError(MemexError):
    """Raised when storage initialization or persistence fails."""


class EmbedderError(MemexError):
    """Raised when embedding generation fails."""


class OptionalDependencyError(MemexError, ImportError):
    """Raised when an optional dependency is required but missing."""


class ImportValidationError(MemexValidationError):
    """Raised when an import file has an invalid or unsafe shape."""
