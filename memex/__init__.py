"""Local-first semantic memory for LLM applications.

The public API is intentionally small. Most applications only need
``Memory``:

    from memex import Memory

    mem = Memory()
    mem.save("The user prefers concise answers.")
    mem.search("How should I answer?")
"""

from memex.core import Memory
from memex.embedders import (
    BaseEmbedder,
    HashEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from memex.models import MemoryRecord, MemoryStats

__all__ = [
    "BaseEmbedder",
    "HashEmbedder",
    "Memory",
    "MemoryRecord",
    "MemoryStats",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
]

__version__ = "0.1.0"
