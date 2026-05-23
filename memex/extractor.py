"""Local fact extraction for ``Memory.learn``."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from re import Pattern
from typing import ClassVar, Protocol

from memex.utils import normalize_text


class BaseExtractor(Protocol):
    """Protocol for durable-fact extractors."""

    def extract(self, user_msg: str, assistant_msg: str) -> list[str]:
        """Extract durable facts worth saving."""


class RuleBasedExtractor:
    """Small local extractor that avoids any inference dependency by default."""

    _PATTERNS: ClassVar[list[Pattern[str]]] = [
        re.compile(r"\bmy name is (?P<value>[^.!?\n]{1,120})", re.IGNORECASE),
        re.compile(r"\bi am (?P<value>[^.!?\n]{1,120})", re.IGNORECASE),
        re.compile(r"\bi'm (?P<value>[^.!?\n]{1,120})", re.IGNORECASE),
        re.compile(r"\bi live in (?P<value>[^.!?\n]{1,120})", re.IGNORECASE),
        re.compile(r"\bi work (?:at|for|on|with) (?P<value>[^.!?\n]{1,120})", re.IGNORECASE),
        re.compile(r"\bi prefer (?P<value>[^.!?\n]{1,120})", re.IGNORECASE),
        re.compile(r"\bi like (?P<value>[^.!?\n]{1,120})", re.IGNORECASE),
        re.compile(r"\bremember that (?P<value>[^.!?\n]{1,160})", re.IGNORECASE),
        re.compile(r"\bmy goal is (?P<value>[^.!?\n]{1,160})", re.IGNORECASE),
    ]

    def __init__(self, *, max_facts: int = 3) -> None:
        self.max_facts = max_facts

    def extract(self, user_msg: str, assistant_msg: str) -> list[str]:
        """Extract facts from the user side of a conversation turn."""

        del assistant_msg
        text = normalize_text(user_msg, max_chars=20_000)
        facts: list[str] = []
        for pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                fact = self._fact_from_match(pattern.pattern, match.group("value"))
                if fact and fact not in facts:
                    facts.append(fact)
                    if len(facts) >= self.max_facts:
                        return facts
        return facts

    def _fact_from_match(self, pattern: str, value: str) -> str:
        cleaned = value.strip(" .!?\n\t")
        if not cleaned:
            return ""
        lower = pattern.lower()
        if "my name is" in lower:
            return f"User's name is {cleaned}."
        if "i live in" in lower:
            return f"User lives in {cleaned}."
        if "i work" in lower:
            return f"User works with {cleaned}."
        if "i prefer" in lower:
            return f"User prefers {cleaned}."
        if "i like" in lower:
            return f"User likes {cleaned}."
        if "remember that" in lower:
            return cleaned[0].upper() + cleaned[1:] + "."
        if "my goal is" in lower:
            return f"User's goal is {cleaned}."
        return f"User is {cleaned}."


class CallableExtractor:
    """Adapter for user-provided extraction callables."""

    def __init__(self, extractor: Callable[[str, str], list[str]]) -> None:
        self.extractor = extractor

    def extract(self, user_msg: str, assistant_msg: str) -> list[str]:
        """Call the wrapped extractor."""

        facts = self.extractor(user_msg, assistant_msg)
        return [normalize_text(fact, max_chars=1_000) for fact in facts[:3]]


class JsonLLMExtractor:
    """Extractor adapter for local or hosted chat callables.

    ``generate`` receives a prompt and must return a JSON array of strings.
    This keeps cloud/model choices outside memex core.
    """

    def __init__(self, generate: Callable[[str], str], *, max_facts: int = 3) -> None:
        self.generate = generate
        self.max_facts = max_facts

    def extract(self, user_msg: str, assistant_msg: str) -> list[str]:
        """Extract facts by asking the configured generator for JSON."""

        prompt = (
            "Extract 0-3 concise facts from this exchange that are worth remembering "
            "long-term. Only output durable preferences, identity, goals, or context. "
            "Return a JSON array of strings, or [] if nothing is worth remembering.\n\n"
            f"User: {user_msg}\nAssistant: {assistant_msg}"
        )
        raw = self.generate(prompt)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        facts: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                facts.append(normalize_text(item, max_chars=1_000))
                if len(facts) >= self.max_facts:
                    break
        return facts
