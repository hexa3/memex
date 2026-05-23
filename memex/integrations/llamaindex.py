"""LlamaIndex integration helpers."""

from __future__ import annotations

import importlib
from typing import Any

from memex.core import Memory


class MemexContext:
    """Adapter for hybrid memex retrieval in LlamaIndex-style pipelines."""

    def __init__(
        self,
        *,
        namespace: str = "default",
        k: int = 5,
        hybrid: bool = True,
        memory: Memory | None = None,
    ) -> None:
        self.memory = memory or Memory(namespace=namespace)
        self.k = k
        self.hybrid = hybrid

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return LlamaIndex-friendly memory nodes with traceability metadata."""

        records = (
            self.memory.hybrid_search(query, k=self.k)
            if self.hybrid
            else self.memory.search(query, k=self.k)
        )
        return [
            {
                "text": record.text,
                "score": record.score,
                "metadata": {
                    **(record.metadata or {}),
                    "memex_id": record.id,
                    "memory_type": record.memory_type,
                    "source_ids": record.source_ids,
                },
            }
            for record in records
        ]

    def as_llamaindex_nodes(self, query: str) -> list[Any]:
        """Return ``TextNode`` objects when LlamaIndex is installed, else dict nodes."""

        nodes = self.retrieve(query)
        try:
            schema = importlib.import_module("llama_index.core.schema")
        except ImportError:
            return nodes
        text_node = schema.TextNode
        return [
            text_node(
                text=node["text"],
                metadata={**node["metadata"], "retrieval_score": node["score"]},
            )
            for node in nodes
        ]

    def inject(self, prompt: str) -> str:
        """Inject relevant memory into a prompt."""

        return self.memory.inject(prompt, k=self.k)
