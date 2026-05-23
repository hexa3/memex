"""Built-in embedders."""

from __future__ import annotations

from importlib.util import find_spec

from memex.embedders.base import BaseEmbedder, ensure_embedder
from memex.embedders.hash import HashEmbedder
from memex.embedders.openai import OpenAIEmbedder
from memex.embedders.sentence import (
    BgeSmallEmbedder,
    MiniLMEmbedder,
    NomicEmbedder,
    SentenceTransformerEmbedder,
)


def create_embedder(name: str | BaseEmbedder = "auto") -> BaseEmbedder:
    """Create a built-in embedder by name.

    ``auto`` prefers MiniLM when sentence-transformers is installed and falls
    back to ``HashEmbedder`` otherwise. Named model embedders raise a helpful
    optional dependency error when their runtime is missing.
    """

    if not isinstance(name, str):
        return ensure_embedder(name)

    key = name.lower().strip()
    if key == "auto":
        if find_spec("sentence_transformers") is not None:
            return MiniLMEmbedder()
        return HashEmbedder()
    if key in {"hash", "test"}:
        return HashEmbedder()
    if key in {"minilm", "mini-lm", "all-minilm-l6-v2"}:
        return MiniLMEmbedder()
    if key in {"bge", "bge-small"}:
        return BgeSmallEmbedder()
    if key == "nomic":
        return NomicEmbedder()
    if key == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"unknown embedder: {name}")


__all__ = [
    "BaseEmbedder",
    "BgeSmallEmbedder",
    "HashEmbedder",
    "MiniLMEmbedder",
    "NomicEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
    "ensure_embedder",
]
