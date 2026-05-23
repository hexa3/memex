# API Reference

## Python

```python
from memex import Memory

mem = Memory(path=None, namespace="default", embedder="auto")
```

### Save

```python
memory_id = mem.save(
    "User prefers concise answers.",
    metadata={"source": "profile"},
    memory_type="semantic",
    importance=1.0,
)
```

### Search

```python
mem.search("concise answers", k=5)
mem.hybrid_search("exact project codename atlas-77", k=5)
mem.recall("how should I answer?")
```

### Learn

```python
facts = mem.learn("My name is Riley. I prefer short answers.", "Got it.")
```

### Summarize

```python
result = mem.summarize(min_sources=8, max_sources=50)
print(result.summary_id)
print(result.source_ids)
```

### Optimize

```python
mem.optimize()
mem.start_cleanup_scheduler()
mem.stop_cleanup_scheduler()
```

### Import And Export

```python
payload = mem.export("memex-export.json")
count = mem.import_from(payload)
```

## TypeScript

```ts
import { Memory } from "memex-ai";

const mem = new Memory({ namespace: "default" });
await mem.save("User prefers local-first tools.", { memoryType: "semantic" });
const results = await mem.hybridSearch("local-first", { k: 3 });
```

## REST

Start the server:

```bash
memex serve
```

Core endpoints:

- `POST /save`
- `GET /search?q=...&hybrid=true`
- `GET /recall?q=...`
- `POST /learn`
- `POST /summarize`
- `GET /export`
- `POST /import`
- `GET /stats`

## LlamaIndex

```python
from memex.integrations.llamaindex import MemexContext

ctx = MemexContext(memory=mem, hybrid=True)
nodes = ctx.retrieve("what user context matters?")
```

## Sync Primitives

```python
from memex.sync import CRDTState, EncryptedSyncCodec

codec = EncryptedSyncCodec.from_passphrase("correct horse battery staple")
envelope = codec.encrypt(b"payload")
payload = codec.decrypt(envelope)
```
