"""Simple benchmark runner that avoids pytest plugin requirements."""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from memex import Memory


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def run(
    count: int,
    queries: int,
    path: Path,
    *,
    use_sqlite_vec: bool,
    batch_size: int,
) -> None:
    mem = Memory(
        path=path,
        embedder="hash",
        use_sqlite_vec=use_sqlite_vec,
        auto_dedupe=False,
        max_memories=count + queries + batch_size,
    )
    prefill_start = time.perf_counter()
    for start_index in range(0, count, batch_size):
        batch = [
            f"memory {index} about local-first sqlite embeddings"
            for index in range(start_index, min(start_index + batch_size, count))
        ]
        mem.save_many(batch)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000

    save_latencies: list[float] = []
    for index in range(queries):
        start = time.perf_counter()
        mem.save(f"measured memory {index} about local-first sqlite embeddings")
        save_latencies.append((time.perf_counter() - start) * 1000)

    search_latencies: list[float] = []
    for index in range(queries):
        start = time.perf_counter()
        mem.search(f"sqlite embeddings {index}", k=5)
        search_latencies.append((time.perf_counter() - start) * 1000)

    print(
        f"count={count} queries={queries} db={path} "
        f"vector_index={mem.stats().vector_index}"
    )
    print(f"prefill_ms={prefill_ms:.3f} batch_size={batch_size}")
    print(
        "save_ms "
        f"p50={statistics.median(save_latencies):.3f} "
        f"p95={percentile(save_latencies, 0.95):.3f}"
    )
    print(
        "search_ms "
        f"p50={statistics.median(search_latencies):.3f} "
        f"p95={percentile(search_latencies, 0.95):.3f}"
    )
    mem.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--no-sqlite-vec", action="store_true")
    args = parser.parse_args()
    use_sqlite_vec = not args.no_sqlite_vec

    if args.db is None:
        with tempfile.TemporaryDirectory() as tmp:
            run(
                args.count,
                args.queries,
                Path(tmp) / "memex.db",
                use_sqlite_vec=use_sqlite_vec,
                batch_size=args.batch_size,
            )
    else:
        run(
            args.count,
            args.queries,
            args.db,
            use_sqlite_vec=use_sqlite_vec,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
