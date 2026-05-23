"""High-level memory API."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from memex.embedders import BaseEmbedder, create_embedder
from memex.events import EventBus, EventHandler
from memex.extractor import BaseExtractor, CallableExtractor, RuleBasedExtractor
from memex.models import MemoryRecord, MemoryStats, SaveRequest, SearchRequest
from memex.storage import SQLiteStorage, StoredMemory
from memex.utils import (
    normalize_namespace,
    normalize_text,
    safe_json_loads,
    utc_timestamp,
    validate_metadata,
)

MemoryRecords = list[MemoryRecord]
FloatVector = list[float]


class Memory:
    """Persistent, semantically searchable local memory.

    Parameters are deliberately boring: a SQLite path, a namespace, and an
    embedder. The default ``auto`` embedder uses MiniLM when available and a
    deterministic local fallback otherwise, so tests and offline examples run
    without model downloads.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        namespace: str = "default",
        embedder: str | BaseEmbedder = "auto",
        storage: SQLiteStorage | None = None,
        auto_dedupe: bool = True,
        dedupe_threshold: float = 0.92,
        max_memories: int = 10_000,
        extractor: BaseExtractor | Callable[[str, str], list[str]] | None = None,
        use_sqlite_vec: bool = True,
        async_prune: bool = True,
        prune_check_interval: int = 64,
    ) -> None:
        self.namespace = normalize_namespace(namespace)
        self.embedder = create_embedder(embedder)
        self.storage = storage or SQLiteStorage(
            path,
            dimension=self.embedder.dimension,
            use_sqlite_vec=use_sqlite_vec,
        )
        self.auto_dedupe = auto_dedupe
        self.dedupe_threshold = dedupe_threshold
        self.max_memories = max_memories
        self.async_prune = async_prune
        self.prune_check_interval = max(1, prune_check_interval)
        self._writes_since_prune_check: dict[str, int] = {}
        self._prune_lock = threading.Lock()
        self._prune_running = False
        self.extractor: BaseExtractor
        if extractor is None:
            self.extractor = RuleBasedExtractor()
        elif callable(extractor) and not hasattr(extractor, "extract"):
            self.extractor = CallableExtractor(extractor)
        else:
            self.extractor = cast(BaseExtractor, extractor)
        self.events = EventBus()

    def save(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        ttl_days: int | None = None,
        importance: float = 1.0,
        namespace: str | None = None,
    ) -> str:
        """Store a memory and return its id."""

        request = SaveRequest(
            text=normalize_text(text),
            metadata=validate_metadata(metadata),
            ttl_days=ttl_days,
            importance=importance,
            namespace=normalize_namespace(namespace or self.namespace),
        )
        embedding = self.embedder.embed(request.text[:20_000])
        now = utc_timestamp()
        memory_id = str(uuid.uuid4())
        self.storage.save(
            StoredMemory(
                id=memory_id,
                namespace=request.namespace,
                text=request.text,
                embedding=embedding,
                metadata=request.metadata,
                importance=request.importance,
                created_at=now,
                accessed_at=now,
                ttl_days=request.ttl_days,
            )
        )
        if self.auto_dedupe:
            self._dedupe(memory_id, request.namespace, embedding)
        self._prune_if_needed(request.namespace)
        self.events.emit("memory_saved", {"id": memory_id, "namespace": request.namespace})
        return memory_id

    def save_many(
        self,
        texts: list[str],
        *,
        metadata: dict[str, Any] | None = None,
        ttl_days: int | None = None,
        importance: float = 1.0,
        namespace: str | None = None,
    ) -> list[str]:
        """Store many memories with one embedding batch and one SQLite transaction."""

        if not texts:
            return []
        target_namespace = normalize_namespace(namespace or self.namespace)
        validated_metadata = validate_metadata(metadata)
        requests = [
            SaveRequest(
                text=normalize_text(text),
                metadata=validated_metadata,
                ttl_days=ttl_days,
                importance=importance,
                namespace=target_namespace,
            )
            for text in texts
        ]
        embeddings = self.embedder.embed_batch([request.text[:20_000] for request in requests])
        now = utc_timestamp()
        ids = [str(uuid.uuid4()) for _ in requests]
        stored = [
            StoredMemory(
                id=memory_id,
                namespace=request.namespace,
                text=request.text,
                embedding=embedding,
                metadata=request.metadata,
                importance=request.importance,
                created_at=now,
                accessed_at=now,
                ttl_days=request.ttl_days,
            )
            for memory_id, request, embedding in zip(ids, requests, embeddings, strict=True)
        ]
        self.storage.save_many(stored)
        if self.auto_dedupe:
            for memory_id, request, embedding in zip(ids, requests, embeddings, strict=True):
                self._dedupe(memory_id, request.namespace, embedding)
        self._prune_if_needed(target_namespace, write_count=len(stored))
        for memory_id in ids:
            self.events.emit("memory_saved", {"id": memory_id, "namespace": target_namespace})
        return ids

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
        include_embedding: bool = False,
    ) -> list[MemoryRecord]:
        """Search memories by semantic similarity."""

        request = SearchRequest(
            query=normalize_text(query, max_chars=20_000),
            k=k,
            threshold=threshold,
            filters=validate_metadata(filters) if filters else None,
            namespace=normalize_namespace(namespace or self.namespace),
        )
        embedding = self.embedder.embed(request.query[:20_000])
        return self.storage.search(
            namespace=request.namespace,
            embedding=embedding,
            k=request.k,
            threshold=request.threshold,
            filters=request.filters,
            include_embedding=include_embedding,
        )

    def recall(
        self,
        query: str,
        *,
        threshold: float = 0.0,
        namespace: str | None = None,
    ) -> str | None:
        """Return the best matching memory text, if any."""

        results = self.search(query, k=1, threshold=threshold, namespace=namespace)
        return results[0].text if results else None

    def learn(
        self,
        user_msg: str,
        assistant_msg: str,
        *,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Extract durable facts from a conversation turn and save them."""

        facts = self.extractor.extract(user_msg, assistant_msg)
        saved: list[str] = []
        for fact in facts[:3]:
            self.save(
                fact,
                namespace=namespace,
                metadata={"source": "learn", **(metadata or {})},
            )
            saved.append(fact)
        return saved

    def inject(self, prompt: str, *, k: int = 5, prefix: str = "[Memory context]") -> str:
        """Return ``prompt`` with relevant memories prepended."""

        memories = self.search(prompt, k=k)
        if not memories:
            return prompt
        block = "\n".join(f"- {memory.text}" for memory in memories)
        return f"{prefix}\n{block}\n\n{prompt}"

    def inject_system(self, prompt: str, *, k: int = 5) -> str:
        """Return a system prompt fragment containing relevant memory."""

        memories = self.search(prompt, k=k)
        if not memories:
            return "You are a helpful assistant."
        block = "\n".join(f"- {memory.text}" for memory in memories)
        return (
            "You are a helpful assistant. You have the following memory about this user:\n"
            f"{block}"
        )

    def forget(
        self,
        query: str,
        *,
        k: int = 5,
        threshold: float = 0.75,
        namespace: str | None = None,
    ) -> int:
        """Forget memories matching a query."""

        results = self.search(query, k=k, threshold=threshold, namespace=namespace)
        deleted = self.storage.soft_delete(record.id for record in results)
        if deleted:
            self.events.emit(
                "memory_deleted",
                {"count": deleted, "namespace": normalize_namespace(namespace or self.namespace)},
            )
        return deleted

    def clear(self, *, namespace: str | None = None) -> int:
        """Soft-delete all memories in a namespace."""

        deleted = self.storage.clear(namespace=namespace or self.namespace)
        self.events.emit(
            "memory_cleared",
            {"count": deleted, "namespace": namespace or self.namespace},
        )
        return deleted

    def list(
        self,
        *,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_embedding: bool = False,
    ) -> MemoryRecords:
        """List memories in reverse creation order."""

        return self.storage.list(
            namespace=namespace or self.namespace,
            limit=limit,
            offset=offset,
            include_embedding=include_embedding,
        )

    def stats(self, *, namespace: str | None = None) -> MemoryStats:
        """Return storage statistics."""

        return self.storage.stats(namespace=namespace or self.namespace)

    def export(
        self,
        path: str | Path | None = None,
        *,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Export memories as JSON-compatible data and optionally write a file."""

        records = self.storage.export_records(namespace=namespace or self.namespace)
        payload = {
            "version": 1,
            "dimension": self.embedder.dimension,
            "records": [
                record.model_dump(mode="json", exclude_none=True)
                for record in records
            ],
        }
        if path is not None:
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def import_from(
        self,
        path_or_payload: str | Path | dict[str, Any],
        *,
        namespace: str | None = None,
        replace: bool = False,
    ) -> int:
        """Import memories from a JSON export.

        Records without embeddings are embedded during import. Records with
        embeddings must match the active embedder dimension.
        """

        payload: dict[str, Any]
        if isinstance(path_or_payload, dict):
            payload = path_or_payload
        else:
            raw = Path(path_or_payload).expanduser().read_text(encoding="utf-8")
            parsed = safe_json_loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("import file must contain a JSON object")
            payload = parsed
        records_raw = payload.get("records")
        if not isinstance(records_raw, list):
            raise ValueError("import payload must contain a records array")
        imported_records: MemoryRecords = []
        target_namespace = normalize_namespace(namespace) if namespace else None
        for item in records_raw:
            if not isinstance(item, dict):
                raise ValueError("each imported record must be an object")
            record = MemoryRecord.model_validate(item)
            if target_namespace:
                record.namespace = target_namespace
            if record.embedding is None:
                record.embedding = self.embedder.embed(record.text[:20_000])
            imported_records.append(record)
        return self.storage.import_records(imported_records, replace=replace)

    def on(self, event: str, handler: EventHandler) -> None:
        """Register an event hook."""

        self.events.on(event, handler)

    def close(self) -> None:
        """Close the storage connection."""

        self.storage.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown defensive path
        with suppress(Exception):
            self.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _dedupe(self, memory_id: str, namespace: str, embedding: FloatVector) -> None:
        matches = self.storage.search(
            namespace=namespace,
            embedding=embedding,
            k=3,
            threshold=self.dedupe_threshold,
            exclude_ids={memory_id},
            update_access=False,
        )
        if not matches:
            return
        closest = matches[0]
        if closest.similarity is not None and closest.similarity >= self.dedupe_threshold:
            self.storage.soft_delete([closest.id])

    def _prune_if_needed(self, namespace: str, *, write_count: int = 1) -> None:
        if self.max_memories <= 0:
            return
        writes = self._writes_since_prune_check.get(namespace, 0) + write_count
        self._writes_since_prune_check[namespace] = writes
        if writes < self.prune_check_interval:
            return
        self._writes_since_prune_check[namespace] = 0
        if self.storage.count(namespace=namespace) <= self.max_memories:
            return
        if not self.async_prune:
            self.storage.prune(namespace=namespace, max_memories=self.max_memories)
            return
        with self._prune_lock:
            if self._prune_running:
                return
            self._prune_running = True
        thread = threading.Thread(
            target=self._run_prune,
            args=(namespace,),
            name=f"memex-prune-{namespace}",
            daemon=True,
        )
        thread.start()

    def _run_prune(self, namespace: str) -> None:
        try:
            self.storage.prune(namespace=namespace, max_memories=self.max_memories)
        finally:
            with self._prune_lock:
                self._prune_running = False
