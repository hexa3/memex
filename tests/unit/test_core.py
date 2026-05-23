from memex import Memory
from memex.embedders import HashEmbedder


def test_memory_save_search_recall(tmp_path) -> None:
    mem = Memory(path=tmp_path / "memex.db", embedder=HashEmbedder(), use_sqlite_vec=False)

    memory_id = mem.save("User prefers concise Python examples.")
    results = mem.search("Python examples", k=1)

    assert memory_id
    assert results[0].text == "User prefers concise Python examples."
    assert mem.recall("Python examples") == "User prefers concise Python examples."


def test_memory_save_many(tmp_path) -> None:
    mem = Memory(
        path=tmp_path / "memex.db",
        embedder="hash",
        use_sqlite_vec=False,
        auto_dedupe=False,
    )

    ids = mem.save_many(["User likes SQLite.", "User likes local-first apps."])

    assert len(ids) == 2
    assert mem.recall("local-first apps") == "User likes local-first apps."


def test_namespace_isolation(tmp_path) -> None:
    mem = Memory(path=tmp_path / "memex.db", embedder="hash", namespace="a", use_sqlite_vec=False)
    mem.save("Namespace A memory")
    mem.save("Namespace B memory", namespace="b")

    assert mem.search("Namespace B", namespace="b", k=1)[0].text == "Namespace B memory"
    assert all(record.namespace == "a" for record in mem.search("Namespace", namespace="a", k=5))


def test_dedupe_keeps_newer_memory(tmp_path) -> None:
    mem = Memory(path=tmp_path / "memex.db", embedder="hash", use_sqlite_vec=False)

    first = mem.save("User likes dark mode.")
    second = mem.save("User likes dark mode.")

    assert first != second
    assert mem.storage.get(first) is None
    assert mem.storage.get(second) is not None


def test_learn_saves_facts(tmp_path) -> None:
    mem = Memory(path=tmp_path / "memex.db", embedder="hash", use_sqlite_vec=False)

    facts = mem.learn("My name is Riley. I prefer short answers.", "Understood.")

    assert "User's name is Riley." in facts
    assert mem.search("Riley", k=1)


def test_export_import_roundtrip(tmp_path) -> None:
    source = Memory(path=tmp_path / "source.db", embedder="hash", use_sqlite_vec=False)
    source.save("User likes local-first software.")
    payload = source.export()

    target = Memory(path=tmp_path / "target.db", embedder="hash", use_sqlite_vec=False)
    imported = target.import_from(payload)

    assert imported == 1
    assert target.recall("local-first software") == "User likes local-first software."
