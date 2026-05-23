"""Embedder protocols and helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol implemented by embedding providers."""

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""

    def embed(self, text: str) -> list[float]:
        """Embed one string."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""


def ensure_embedder(embedder: object) -> BaseEmbedder:
    """Validate that an object implements the embedder protocol."""

    if not isinstance(embedder, BaseEmbedder):
        raise TypeError("embedder must implement embed(), embed_batch(), and dimension")
    return embedder
