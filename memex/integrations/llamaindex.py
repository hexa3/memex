"""LlamaIndex integration helpers."""

from __future__ import annotations

from typing import Any

from memex.core import Memory


class MemexContext:
    """Small adapter for injecting memex context into LlamaIndex prompts."""

    def __init__(
        self,
        *,
        namespace: str = "default",
        k: int = 5,
        memory: Memory | None = None,
    ) -> None:
        self.memory = memory or Memory(namespace=namespace)
        self.k = k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return LlamaIndex-friendly memory nodes."""

        return [
            {
                "text": record.text,
                "score": record.score,
                "metadata": record.metadata or {},
            }
            for record in self.memory.search(query, k=self.k)
        ]

    def inject(self, prompt: str) -> str:
        """Inject relevant memory into a prompt."""

        return self.memory.inject(prompt, k=self.k)
