# Architecture

memex is a local-first AI memory system with a small public API and optional advanced layers.

## System Diagram

```text
User Apps
  |-- Python SDK
  |-- TypeScript SDK
  |-- CLI
  |-- Browser Extension
        |
        v
Memory API
  |-- Save / Learn
  |-- Semantic Search
  |-- Hybrid Search
  |-- Summarize / Optimize
        |
        v
Storage + Retrieval
  |-- SQLite metadata and embeddings
  |-- sqlite-vec when available
  |-- Python scan fallback
  |-- Optional Rust core acceleration
        |
        v
Optional Sync
  |-- CRDT merge
  |-- encrypted payload envelopes
  |-- user-controlled transport
```

## Memory Hierarchy

Every memory has a `memory_type`:

- `short_term`: temporary context or low-durability state.
- `long_term`: durable facts and preferences.
- `episodic`: events, conversations, sessions, and project moments.
- `semantic`: distilled user or domain knowledge.
- `summary`: compressed memories with `source_ids` traceability.

## Retrieval

`search()` performs semantic retrieval. `hybrid_search()` combines:

1. Vector similarity from the configured embedder.
2. Keyword fallback for exact identifiers, names, and codes.
3. Final score ranking with importance, recency, and access count.

sqlite-vec is used when available. If loading fails, memex falls back to a deterministic Python scan.

## Summarization

`summarize()` creates a semantic summary from source memories. The new summary stores:

- `memory_type="summary"`
- `source_ids=[...]`
- metadata with `traceability.source_ids`

By default, source memories are preserved. Callers can pass `delete_sources=True` to replace source records with the summary.

## Sync And Security

Sync is optional and disabled by default. The sync primitives use:

- ChaCha20-Poly1305 encrypted envelopes through the `sync` extra.
- CRDT merge semantics for offline-first conflict handling.
- No plaintext cloud storage assumptions.

Transports are intentionally outside the core package. This keeps self-hosted, peer-to-peer, and hosted backends interchangeable.

## Rust Core

The `rust-core` crate contains performance-sensitive routines:

- cosine ranking and top-k retrieval
- extractive summary helpers
- sync diff helpers

It exposes optional `python` and `node` feature flags for PyO3 and N-API bindings. Python automatically falls back to pure Python when the Rust module is not installed.
