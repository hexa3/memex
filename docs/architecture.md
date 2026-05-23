# Architecture

memex is built around four narrow layers:

1. `Memory`: public user API.
2. `BaseEmbedder`: pluggable embedding interface.
3. `SQLiteStorage`: persistence, migrations, search, import/export.
4. `BaseExtractor`: optional durable-fact extraction for `learn`.

The core package imports only lightweight dependencies at startup. Heavy embedding and server frameworks are imported lazily at the boundary that uses them.

## Storage

The SQLite schema stores text, metadata JSON, importance, TTL, access counters, and a float32 embedding blob. sqlite-vec is loaded if installed and supported on the platform. If loading fails, search falls back to scanning active rows and computing cosine similarity in Python.

## Consistency

`save()` writes the memory row and vector row in one SQLite transaction. Deletion is soft by default so imports and exports can remain conservative and recovery-friendly.

## Pruning

When a namespace exceeds its configured cap, memex removes the lowest effective-importance records. Effective importance combines the saved score, recency decay, and access count.
