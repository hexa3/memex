"""Runtime adapter for the optional Rust core engine."""

from __future__ import annotations


class CoreEngine:
    """Small facade that uses Rust acceleration when installed."""

    def __init__(self) -> None:
        try:
            import memex_core  # type: ignore[import-not-found]
        except ImportError:
            self._rust = None
        else:
            self._rust = memex_core

    @property
    def accelerated(self) -> bool:
        """Return whether Rust acceleration is active."""

        return self._rust is not None

    def summarize(self, texts: list[str], *, max_sentences: int) -> str:
        """Summarize text with Rust when available, else a Python fallback."""

        if self._rust is not None:
            return str(self._rust.summarize(texts, max_sentences))
        sentences: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for chunk in text.replace("\n", " ").split("."):
                sentence = chunk.strip()
                if not sentence:
                    continue
                key = " ".join(sentence.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                sentences.append(sentence.rstrip(".") + ".")
                if len(sentences) >= max_sentences:
                    return " ".join(sentences)
        return " ".join(sentences)
