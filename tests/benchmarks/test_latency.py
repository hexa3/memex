import time

from memex import Memory


def test_hash_backend_smoke_latency(tmp_path) -> None:
    mem = Memory(
        path=tmp_path / "memex.db",
        embedder="hash",
        use_sqlite_vec=False,
        auto_dedupe=False,
    )
    for index in range(100):
        mem.save(f"memory {index} about local sqlite search")

    start = time.perf_counter()
    mem.search("sqlite search", k=5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 200
