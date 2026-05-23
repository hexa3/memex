"""LangChain integration."""

from __future__ import annotations

from typing import Any

from memex.core import Memory
from memex.errors import OptionalDependencyError

try:
    from langchain.memory.chat_memory import BaseChatMemory
except ImportError:  # pragma: no cover - exercised only when optional dep exists
    BaseChatMemory = object


class MemexMemory(BaseChatMemory):  # type: ignore[misc]
    """LangChain ``BaseChatMemory`` backed by memex."""

    memory_key: str = "history"

    def __init__(self, *, namespace: str = "default", k: int = 5, **kwargs: Any) -> None:
        if BaseChatMemory is object:
            raise OptionalDependencyError(
                "LangChain integration requires: pip install 'memex-ai[langchain]'"
            )
        super().__init__(**kwargs)
        self.memex = Memory(namespace=namespace)
        self.k = k

    @property
    def memory_variables(self) -> list[str]:
        """Variables injected by this memory."""

        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Search memex for context relevant to the latest input."""

        query = " ".join(str(value) for value in inputs.values())
        memories = self.memex.search(query, k=self.k)
        block = "\n".join(memory.text for memory in memories)
        return {self.memory_key: block}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Learn from a LangChain turn."""

        user_msg = " ".join(str(value) for value in inputs.values())
        assistant_msg = " ".join(str(value) for value in outputs.values())
        self.memex.learn(user_msg, assistant_msg)

    def clear(self) -> None:
        """Clear the backing namespace."""

        self.memex.clear()
