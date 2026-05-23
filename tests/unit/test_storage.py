from memex.embedders import HashEmbedder
from memex.storage import SQLiteStorage, StoredMemory
from memex.utils import utc_timestamp


def _stored(memory_id: str, text: str, namespace: str = "default") -> StoredMemory:
    embedder = HashEmbedder()
    now = utc_timestamp()
    return StoredMemory(
        id=memory_id,
        namespace=namespace,
        text=text,
        embedding=embedder.embed(text),
        metadata={"kind": "test"},
        importance=1.0,
        created_at=now,
        accessed_at=now,
        ttl_days=None,
    )


def test_storage_save_and_search(tmp_path) -> None:
    embedder = HashEmbedder()
    storage = SQLiteStorage(
        tmp_path / "memex.db",
        dimension=embedder.dimension,
        use_sqlite_vec=False,
    )
    storage.save(_stored("one", "User prefers dark mode"))

    results = storage.search(
        namespace="default",
        embedding=embedder.embed("dark mode preference"),
        k=1,
        threshold=-1.0,
    )

    assert results[0].id == "one"
    assert results[0].access_count == 0
    assert storage.get("one").access_count == 1


def test_storage_metadata_filters(tmp_path) -> None:
    embedder = HashEmbedder()
    storage = SQLiteStorage(
        tmp_path / "memex.db",
        dimension=embedder.dimension,
        use_sqlite_vec=False,
    )
    storage.save(_stored("one", "User prefers dark mode"))

    assert storage.search(
        namespace="default",
        embedding=embedder.embed("dark mode"),
        k=1,
        filters={"kind": "test"},
    )
    assert not storage.search(
        namespace="default",
        embedding=embedder.embed("dark mode"),
        k=1,
        filters={"kind": "other"},
    )


def test_storage_prune(tmp_path) -> None:
    embedder = HashEmbedder()
    storage = SQLiteStorage(
        tmp_path / "memex.db",
        dimension=embedder.dimension,
        use_sqlite_vec=False,
    )
    for index in range(5):
        storage.save(_stored(str(index), f"memory {index}"))

    deleted = storage.prune(namespace="default", max_memories=3)

    assert deleted >= 2
    assert storage.count(namespace="default") <= 3
