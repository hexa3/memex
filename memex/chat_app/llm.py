"""LLM integration for the Memex chat demo."""

from __future__ import annotations

import os
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from memex.models import MemoryRecord


@dataclass(frozen=True)
class ChatMessage:
    """One chat message sent to the LLM layer."""

    role: str
    content: str


class ChatLLM(Protocol):
    """Protocol for streaming chat providers."""

    @property
    def provider(self) -> str:
        """Return provider name."""

    def stream(
        self,
        *,
        user_message: str,
        history: list[ChatMessage],
        memories: list[MemoryRecord],
    ) -> Iterable[str]:
        """Yield assistant text chunks."""


class LocalDemoLLM:
    """Small local responder that makes memory influence visible without API keys."""

    provider = "local-demo"

    def stream(
        self,
        *,
        user_message: str,
        history: list[ChatMessage],
        memories: list[MemoryRecord],
    ) -> Iterable[str]:
        del history
        memory_lines = [memory.text for memory in memories[:3]]
        if memory_lines:
            intro = "I found relevant memory before answering: "
            memory_text = "; ".join(memory_lines)
            response = (
                f"{intro}{memory_text}.\n\n"
                f"With that in mind, here is a practical answer to your message: "
                f"{user_message.strip()}"
            )
        else:
            response = (
                "I do not have a relevant memory for this yet, so I will answer from the current "
                f"conversation: {user_message.strip()}"
            )
        response += (
            "\n\nIf you share stable preferences, goals, or facts about yourself, I will remember "
            "them and show them in the sidebar."
        )
        yield from _chunk_text(response)


class OpenAIChatLLM:
    """OpenAI streaming chat provider used when configured."""

    provider = "openai"

    def __init__(self, *, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any | None = None

    def stream(
        self,
        *,
        user_message: str,
        history: list[ChatMessage],
        memories: list[MemoryRecord],
    ) -> Iterable[str]:
        client = self._load_client()
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": _system_prompt(memories),
            }
        ]
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in history[-12:]
        )
        messages.append({"role": "user", "content": user_message})
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            stream=True,
        )
        for event in stream:
            chunk = event.choices[0].delta.content
            if chunk:
                yield str(chunk)

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install OpenAI support with: pip install 'memex-ai[openai]'"
            ) from exc
        self._client = OpenAI(api_key=self.api_key)
        return self._client


def create_llm(provider: str = "auto") -> ChatLLM:
    """Create the configured LLM provider with a no-config local fallback."""

    selected = provider.strip().lower()
    if selected in {"auto", "openai"} and os.environ.get("OPENAI_API_KEY"):
        return OpenAIChatLLM(model=os.environ.get("MEMEX_CHAT_MODEL", "gpt-4.1-mini"))
    return LocalDemoLLM()


def _system_prompt(memories: list[MemoryRecord]) -> str:
    memory_block = "\n".join(f"- [{memory.memory_type}] {memory.text}" for memory in memories)
    if not memory_block:
        memory_block = "- No relevant memory found."
    return textwrap.dedent(
        f"""
        You are a concise, helpful assistant in a Memex demo app.
        Use the provided memories when relevant, but never claim a memory was used if it was not.
        Mention remembered preferences naturally and briefly.

        Relevant memories:
        {memory_block}
        """
    ).strip()


def _chunk_text(text: str, *, size: int = 18) -> Iterable[str]:
    words = text.split(" ")
    for index in range(0, len(words), size):
        yield " ".join(words[index : index + size]) + (" " if index + size < len(words) else "")
