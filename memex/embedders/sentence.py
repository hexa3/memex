"""Lazy sentence-transformers embedder implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memex.errors import EmbedderError, OptionalDependencyError
from memex.utils import default_data_dir, normalized


class SentenceTransformerEmbedder:
    """Local sentence-transformers backend loaded on first embedding call."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        dimension: int = 384,
        cache_folder: str | Path | None = None,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self.cache_folder = (
            Path(cache_folder).expanduser() if cache_folder else default_data_dir() / "models"
        )
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""

        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch using sentence-transformers."""

        if not texts:
            return []
        model = self._load()
        try:
            encoded = model.encode(
                texts,
                convert_to_numpy=False,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
        except TypeError:
            encoded = model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
        vectors = [[float(value) for value in vector] for vector in encoded]
        if self.normalize_embeddings:
            vectors = [normalized(vector) for vector in vectors]
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbedderError(
                    f"{self.model_name} returned dimension {len(vector)}, "
                    f"expected {self._dimension}"
                )
        return vectors

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                "sentence-transformers is required for this embedder. "
                "Install with: pip install 'memex-ai[local]'"
            ) from exc
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"cache_folder": str(self.cache_folder)}
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model


class MiniLMEmbedder(SentenceTransformerEmbedder):
    """all-MiniLM-L6-v2 embedder."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            **kwargs,
        )


class BgeSmallEmbedder(SentenceTransformerEmbedder):
    """BAAI/bge-small-en-v1.5 embedder."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("BAAI/bge-small-en-v1.5", dimension=384, **kwargs)


class NomicEmbedder(SentenceTransformerEmbedder):
    """nomic-embed-text-v1 embedder."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("nomic-ai/nomic-embed-text-v1", dimension=768, **kwargs)
