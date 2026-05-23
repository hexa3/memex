"""SQLite storage engine with optional sqlite-vec acceleration."""

from __future__ import annotations

import heapq
import importlib
import math
import sqlite3
import threading
from builtins import list as builtin_list
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memex.errors import ImportValidationError, StorageError
from memex.models import MemoryRecord, MemoryStats, datetime_from_timestamp
from memex.utils import (
    cosine_similarity,
    default_db_path,
    metadata_from_json,
    metadata_matches,
    metadata_to_json,
    normalize_namespace,
    pack_embedding,
    secure_file,
    secure_parent_dir,
    unpack_embedding,
    utc_timestamp,
    validate_embedding,
    validate_metadata,
)

FloatVector = builtin_list[float]
MemoryRecordList = builtin_list[MemoryRecord]
MetadataDict = dict[str, Any]
_MISSING = object()


@dataclass(frozen=True)
class StoredMemory:
    """Input shape used by ``SQLiteStorage.save``."""

    id: str
    namespace: str
    text: str
    embedding: FloatVector
    metadata: MetadataDict | None
    importance: float
    created_at: int
    accessed_at: int
    ttl_days: int | None


class SQLiteStorage:
    """Persistence layer for memories and vector blobs."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        dimension: int,
        use_sqlite_vec: bool = True,
        timeout: float = 30.0,
        decay_lambda: float = 0.01,
    ) -> None:
        self.path = secure_parent_dir(Path(path).expanduser() if path else default_db_path())
        self.dimension = dimension
        self.use_sqlite_vec = use_sqlite_vec
        self.timeout = timeout
        self.decay_lambda = decay_lambda
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._sqlite_vec_available = False

    @property
    def vector_index(self) -> str:
        """Return the active vector index mode."""

        return "sqlite-vec" if self._sqlite_vec_available else "python-scan"

    def close(self) -> None:
        """Close the SQLite connection."""

        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown defensive path
        with suppress(Exception):
            self.close()

    def connect(self) -> sqlite3.Connection:
        """Return a configured SQLite connection, opening it if needed."""

        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.path,
                    timeout=self.timeout,
                    isolation_level=None,
                    check_same_thread=False,
                )
                self._connection.row_factory = sqlite3.Row
                self._configure(self._connection)
                self._load_sqlite_vec(self._connection)
                self._migrate(self._connection)
                secure_file(self.path)
            return self._connection

    def save(self, memory: StoredMemory) -> None:
        """Insert or replace a memory and its vector."""

        self.save_many([memory])

    def save_many(self, memories: Iterable[StoredMemory]) -> int:
        """Insert or replace multiple memories in one transaction."""

        memory_list = builtin_list(memories)
        if not memory_list:
            return 0
        with self._lock:
            conn = self.connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for memory in memory_list:
                    embedding_blob = pack_embedding(memory.embedding, dimension=self.dimension)
                    metadata_json = metadata_to_json(validate_metadata(memory.metadata))
                    namespace = normalize_namespace(memory.namespace)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO memories (
                            id, namespace, text, embedding, metadata, importance,
                            created_at, accessed_at, access_count, ttl_days, is_deleted
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0)
                        """,
                        (
                            memory.id,
                            namespace,
                            memory.text,
                            embedding_blob,
                            metadata_json,
                            memory.importance,
                            memory.created_at,
                            memory.accessed_at,
                            memory.ttl_days,
                        ),
                    )
                    if self._sqlite_vec_available:
                        self._insert_vector(conn, memory.id, embedding_blob)
                conn.execute("COMMIT")
            except sqlite3.DatabaseError as exc:
                conn.execute("ROLLBACK")
                raise StorageError(f"failed to save memory: {exc}") from exc
        return len(memory_list)

    def get(self, memory_id: str, *, include_embedding: bool = False) -> MemoryRecord | None:
        """Fetch one memory by id."""

        row = self.connect().execute(
            "SELECT * FROM memories WHERE id = ? AND is_deleted = 0",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row, include_embedding=include_embedding)

    def search(
        self,
        *,
        namespace: str,
        embedding: FloatVector,
        k: int,
        threshold: float = 0.0,
        filters: MetadataDict | None = None,
        exclude_ids: set[str] | None = None,
        include_embedding: bool = False,
        update_access: bool = True,
    ) -> MemoryRecordList:
        """Return top-k memories by cosine similarity weighted by importance."""

        validate_embedding(embedding, dimension=self.dimension)
        namespace = normalize_namespace(namespace)
        filters = validate_metadata(filters) if filters else None
        exclude_ids = exclude_ids or set()
        with self._lock:
            conn = self.connect()
            if self._sqlite_vec_available and not filters:
                try:
                    records = self._search_sqlite_vec(
                        conn,
                        namespace=namespace,
                        embedding=embedding,
                        k=max(k + len(exclude_ids), k),
                        threshold=threshold,
                        exclude_ids=exclude_ids,
                        include_embedding=include_embedding,
                    )
                    records = records[:k]
                    if update_access:
                        self.mark_accessed([record.id for record in records])
                    return records
                except sqlite3.DatabaseError:
                    self._sqlite_vec_available = False
            records = self._search_python(
                conn,
                namespace=namespace,
                embedding=embedding,
                k=k,
                threshold=threshold,
                filters=filters,
                exclude_ids=exclude_ids,
                include_embedding=include_embedding,
            )
            if update_access:
                self.mark_accessed([record.id for record in records])
            return records

    def list(
        self,
        *,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_embedding: bool = False,
    ) -> MemoryRecordList:
        """List active memories ordered by creation time descending."""

        conn = self.connect()
        if namespace is None:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE is_deleted = 0
                  AND (ttl_days IS NULL OR created_at + ttl_days * 86400 > unixepoch())
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE namespace = ? AND is_deleted = 0
                  AND (ttl_days IS NULL OR created_at + ttl_days * 86400 > unixepoch())
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (normalize_namespace(namespace), limit, offset),
            ).fetchall()
        return [self._row_to_record(row, include_embedding=include_embedding) for row in rows]

    def mark_accessed(self, ids: Iterable[str]) -> None:
        """Increment access counters for memory ids."""

        ids = list(dict.fromkeys(ids))
        if not ids:
            return
        now = utc_timestamp()
        placeholders = ", ".join("?" for _ in ids)
        self.connect().execute(
            f"""
            UPDATE memories
            SET accessed_at = ?, access_count = access_count + 1
            WHERE id IN ({placeholders})
            """,
            [now, *ids],
        )

    def soft_delete(self, ids: Iterable[str]) -> int:
        """Soft-delete memory ids and remove vector-index entries when possible."""

        ids = list(dict.fromkeys(ids))
        if not ids:
            return 0
        with self._lock:
            conn = self.connect()
            conn.executemany(
                "UPDATE memories SET is_deleted = 1 WHERE id = ?",
                [(item,) for item in ids],
            )
            if self._sqlite_vec_available:
                try:
                    conn.executemany(
                        "DELETE FROM memory_vectors WHERE id = ?",
                        [(item,) for item in ids],
                    )
                except sqlite3.DatabaseError:
                    self._sqlite_vec_available = False
            return len(ids)

    def clear(self, *, namespace: str | None = None) -> int:
        """Soft-delete all memories, optionally scoped to a namespace."""

        conn = self.connect()
        if namespace is None:
            ids = [
                row["id"]
                for row in conn.execute("SELECT id FROM memories WHERE is_deleted = 0")
            ]
        else:
            ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM memories WHERE namespace = ? AND is_deleted = 0",
                    (normalize_namespace(namespace),),
                )
            ]
        return self.soft_delete(ids)

    def count(self, *, namespace: str | None = None, include_deleted: bool = False) -> int:
        """Count memories."""

        where = []
        params: builtin_list[Any] = []
        if namespace is not None:
            where.append("namespace = ?")
            params.append(normalize_namespace(namespace))
        if not include_deleted:
            where.append("is_deleted = 0")
        clause = " WHERE " + " AND ".join(where) if where else ""
        row = self.connect().execute(
            f"SELECT COUNT(*) AS count FROM memories{clause}",
            params,
        ).fetchone()
        return int(row["count"])

    def stats(self, *, namespace: str | None = None) -> MemoryStats:
        """Return storage stats."""

        conn = self.connect()
        if namespace is None:
            count = self.count()
            deleted = self.count(include_deleted=True) - count
        else:
            count = self.count(namespace=namespace)
            deleted = self.count(namespace=namespace, include_deleted=True) - count
        try:
            size_mb = self.path.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0
        conn.execute("PRAGMA quick_check").fetchone()
        return MemoryStats(
            path=str(self.path),
            namespace=namespace,
            count=count,
            deleted_count=deleted,
            size_mb=round(size_mb, 4),
            vector_index=self.vector_index,
            dimension=self.dimension,
        )

    def export_records(
        self,
        *,
        namespace: str | None = None,
        include_deleted: bool = False,
    ) -> MemoryRecordList:
        """Export records including embeddings."""

        params: builtin_list[Any] = []
        where = []
        if namespace is not None:
            where.append("namespace = ?")
            params.append(normalize_namespace(namespace))
        if not include_deleted:
            where.append("is_deleted = 0")
        clause = " WHERE " + " AND ".join(where) if where else ""
        rows = self.connect().execute(
            f"SELECT * FROM memories{clause} ORDER BY created_at ASC",
            params,
        ).fetchall()
        return [self._row_to_record(row, include_embedding=True) for row in rows]

    def import_records(self, records: MemoryRecordList, *, replace: bool = False) -> int:
        """Import validated records."""

        if len(records) > 100_000:
            raise ImportValidationError("refusing to import more than 100000 records at once")
        imported_records: builtin_list[StoredMemory] = []
        for record in records:
            if record.embedding is None:
                raise ImportValidationError(f"record {record.id} is missing an embedding")
            if len(record.embedding) != self.dimension:
                raise ImportValidationError(
                    f"record {record.id} has dimension {len(record.embedding)}, "
                    f"expected {self.dimension}"
            )
            existing = self.get(record.id)
            if existing is not None:
                if not replace:
                    continue
                self.soft_delete([record.id])
            imported_records.append(
                StoredMemory(
                    id=record.id,
                    namespace=record.namespace,
                    text=record.text,
                    embedding=record.embedding,
                    metadata=record.metadata,
                    importance=record.importance,
                    created_at=int(record.created_at.timestamp()),
                    accessed_at=int(record.accessed_at.timestamp()),
                    ttl_days=record.ttl_days,
                )
            )
        return self.save_many(imported_records)

    def prune(self, *, namespace: str, max_memories: int) -> int:
        """Soft-delete the lowest-ranked memories until a namespace is under cap."""

        current = self.count(namespace=namespace)
        if current <= max_memories:
            return 0
        delete_count = max(current - max_memories, max(1, current // 10))
        rows = self.connect().execute(
            """
            SELECT id, importance, created_at, access_count FROM memories
            WHERE namespace = ? AND is_deleted = 0
            """,
            (normalize_namespace(namespace),),
        ).fetchall()
        scored = [
            (
                self._effective_importance(
                    float(row["importance"]),
                    int(row["created_at"]),
                    int(row["access_count"]),
                ),
                row["id"],
            )
            for row in rows
        ]
        ids = [memory_id for _, memory_id in heapq.nsmallest(delete_count, scored)]
        return self.soft_delete(ids)

    def _configure(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA cache_size=-20000")
        conn.execute("PRAGMA mmap_size=134217728")

    def _load_sqlite_vec(self, conn: sqlite3.Connection) -> None:
        if not self.use_sqlite_vec:
            return
        try:
            sqlite_vec = importlib.import_module("sqlite_vec")
        except ImportError:
            return
        try:
            conn.enable_load_extension(True)
            if hasattr(sqlite_vec, "load"):
                sqlite_vec.load(conn)
            elif hasattr(sqlite_vec, "loadable_path"):
                conn.load_extension(sqlite_vec.loadable_path())
            conn.enable_load_extension(False)
            self._sqlite_vec_available = True
        except Exception:
            self._sqlite_vec_available = False
            with suppress(sqlite3.DatabaseError):
                conn.enable_load_extension(False)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                importance REAL NOT NULL DEFAULT 1.0,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                ttl_days INTEGER DEFAULT NULL,
                is_deleted INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(is_deleted)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memex_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO memex_meta(key, value) VALUES('schema_version', '1')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO memex_meta(key, value) VALUES('dimension', ?)",
            (str(self.dimension),),
        )
        if self._sqlite_vec_available:
            try:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors
                    USING vec0(id TEXT, embedding FLOAT[{self.dimension}])
                    """
                )
            except sqlite3.DatabaseError:
                self._sqlite_vec_available = False

    def _insert_vector(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        embedding_blob: bytes,
    ) -> None:
        try:
            conn.execute("DELETE FROM memory_vectors WHERE id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memory_vectors(id, embedding) VALUES (?, ?)",
                (memory_id, embedding_blob),
            )
        except sqlite3.DatabaseError:
         self._sqlite_vec_available = False

    def _search_sqlite_vec(
        self,
        conn: sqlite3.Connection,
        *,
        namespace: str,
        embedding: FloatVector,
        k: int,
        threshold: float,
        exclude_ids: set[str],
        include_embedding: bool,
    ) -> MemoryRecordList:
        embedding_blob = pack_embedding(embedding, dimension=self.dimension)
        rows = conn.execute(
            """
            WITH nearest AS (
                SELECT id, distance
                FROM memory_vectors
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            )
            SELECT m.*, nearest.distance AS distance
            FROM nearest
            JOIN memories m ON m.id = nearest.id
            WHERE m.namespace = ?
              AND m.is_deleted = 0
              AND (m.ttl_days IS NULL OR m.created_at + m.ttl_days * 86400 > unixepoch())
            ORDER BY nearest.distance ASC
            """,
            (embedding_blob, k, namespace),
        ).fetchall()
        records: MemoryRecordList = []
        for row in rows:
            if row["id"] in exclude_ids:
                continue
            distance = max(0.0, float(row["distance"]))
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            if similarity < threshold:
                continue
            record = self._row_to_record(row, include_embedding=include_embedding)
            effective = self._effective_importance(
                record.importance,
                int(row["created_at"]),
                int(row["access_count"]),
            )
            record.similarity = similarity
            record.score = max(0.0, similarity * effective)
            records.append(record)
        records.sort(key=lambda item: item.score or 0.0, reverse=True)
        return records

    def _search_python(
        self,
        conn: sqlite3.Connection,
        *,
        namespace: str,
        embedding: FloatVector,
        k: int,
        threshold: float,
        filters: dict[str, Any] | None,
        exclude_ids: set[str],
        include_embedding: bool,
    ) -> MemoryRecordList:
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE namespace = ?
              AND is_deleted = 0
              AND (ttl_days IS NULL OR created_at + ttl_days * 86400 > unixepoch())
            """,
            (namespace,),
        ).fetchall()
        hits: builtin_list[tuple[float, MemoryRecord]] = []
        for row in rows:
            if row["id"] in exclude_ids:
                continue
            metadata = metadata_from_json(row["metadata"])
            if not metadata_matches(metadata, filters):
                continue
            stored_embedding = unpack_embedding(row["embedding"], dimension=self.dimension)
            similarity = cosine_similarity(embedding, stored_embedding)
            if similarity < threshold:
                continue
            record = self._row_to_record(
                row,
                include_embedding=include_embedding,
                metadata=metadata,
                embedding=stored_embedding if include_embedding else None,
            )
            effective = self._effective_importance(
                record.importance,
                int(row["created_at"]),
                int(row["access_count"]),
            )
            record.similarity = similarity
            record.score = max(0.0, similarity * effective)
            hits.append((record.score, record))
        top = heapq.nlargest(k, hits, key=lambda item: item[0])
        return [record for _, record in top]

    def _row_to_record(
        self,
        row: sqlite3.Row,
        *,
        include_embedding: bool,
        metadata: MetadataDict | None | object = _MISSING,
        embedding: FloatVector | None = None,
    ) -> MemoryRecord:
        parsed_metadata: MetadataDict | None
        if metadata is _MISSING:
            parsed_metadata = metadata_from_json(row["metadata"])
        elif metadata is None or isinstance(metadata, dict):
            parsed_metadata = metadata
        else:
            parsed_metadata = None
        if include_embedding and embedding is None:
            embedding = unpack_embedding(row["embedding"], dimension=self.dimension)
        return MemoryRecord(
            id=str(row["id"]),
            namespace=str(row["namespace"]),
            text=str(row["text"]),
            embedding=embedding if include_embedding else None,
            metadata=parsed_metadata,
            importance=float(row["importance"]),
            created_at=datetime_from_timestamp(int(row["created_at"])),
            accessed_at=datetime_from_timestamp(int(row["accessed_at"])),
            access_count=int(row["access_count"]),
            ttl_days=int(row["ttl_days"]) if row["ttl_days"] is not None else None,
        )

    def _effective_importance(self, importance: float, created_at: int, access_count: int) -> float:
        age_days = max(0.0, (utc_timestamp() - created_at) / 86400.0)
        recency_factor = math.exp(-self.decay_lambda * age_days)
        access_factor = 1.0 + 0.1 * math.log1p(max(0, access_count))
        return max(0.0, importance * recency_factor * access_factor)
