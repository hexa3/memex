# API Reference

## `Memory`

```python
Memory(
    path=None,
    namespace="default",
    embedder="auto",
    auto_dedupe=True,
    dedupe_threshold=0.92,
    max_memories=10_000,
)
```

## Methods

- `save(text, metadata=None, ttl_days=None, importance=1.0) -> str`
- `save_many(texts, metadata=None, ttl_days=None, importance=1.0) -> list[str]`
- `search(query, k=5, threshold=0.0, filters=None) -> list[MemoryRecord]`
- `recall(query, threshold=0.0) -> str | None`
- `learn(user_msg, assistant_msg) -> list[str]`
- `inject(prompt, k=5, prefix="[Memory context]") -> str`
- `inject_system(prompt, k=5) -> str`
- `forget(query, k=5, threshold=0.75) -> int`
- `clear() -> int`
- `list(limit=100, offset=0) -> list[MemoryRecord]`
- `stats() -> MemoryStats`
- `export(path=None) -> dict`
- `import_from(path_or_payload, replace=False) -> int`
