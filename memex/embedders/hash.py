"""Dependency-free deterministic fallback embeddings.

This embedder is designed for tests, examples, and offline zero-dependency use.
It is lexical rather than model-semantic; install ``memex-ai[local]`` for the
MiniLM sentence-transformer backend.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise

from memex.utils import normalized, tokenize


class HashEmbedder:
    """Fast deterministic feature-hashing embedder."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 16:
            raise ValueError("dimension must be at least 16")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""

        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using signed feature hashing over tokens and bigrams."""

        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = tokenize(text)
        features = list(tokens)
        features.extend(f"{left} {right}" for left, right in pairwise(tokens))
        if not features:
            features = [text.lower()]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self._dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            weight = 1.0 + min(len(feature), 24) / 48.0
            vector[index] += sign * weight
        return normalized(vector)
