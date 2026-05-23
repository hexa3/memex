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
from memex.models import MemoryKind, MemoryRecord, MemoryStats, SummaryResult

__all__ = [
    "BaseEmbedder",
    "HashEmbedder",
    "Memory",
    "MemoryKind",
    "MemoryRecord",
    "MemoryStats",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "SummaryResult",
    "create_embedder",
]

__version__ = "0.1.1"
