"""Optional OpenAI embedding backend."""

from __future__ import annotations

import os
from typing import Any

from memex.errors import OptionalDependencyError


class OpenAIEmbedder:
    """OpenAI text-embedding backend.

    This backend is never used implicitly; callers must opt into it.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        dimension: int = 1536,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._dimension = dimension
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = client

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""

        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch through OpenAI's API."""

        client = self._load_client()
        response = client.embeddings.create(model=self.model, input=texts)
        return [[float(value) for value in item.embedding] for item in response.data]

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise OptionalDependencyError("OPENAI_API_KEY is required for OpenAIEmbedder")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OptionalDependencyError(
                "openai is required for OpenAIEmbedder. "
                "Install with: pip install 'memex-ai[openai]'"
            ) from exc
        self._client = OpenAI(api_key=self.api_key)
        return self._client
