<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yourname/memex/main/docs/logo-dark.svg">
  <img alt="memex" src="https://raw.githubusercontent.com/yourname/memex/main/docs/logo-light.svg" height="72">
</picture>

**Persistent, searchable memory for any LLM.**  
Local-first. No server. No API key. One import.

[![PyPI](https://img.shields.io/pypi/v/memex-ai?style=flat-square&color=0f6e56&label=pypi)](https://pypi.org/project/memex-ai/)
[![npm](https://img.shields.io/npm/v/memex-ai?style=flat-square&color=0f6e56&label=npm)](https://www.npmjs.com/package/memex-ai)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://pypi.org/project/memex-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/yourname/memex/ci.yml?style=flat-square&label=tests)](https://github.com/yourname/memex/actions)
[![Stars](https://img.shields.io/github/stars/yourname/memex?style=flat-square)](https://github.com/yourname/memex/stargazers)

</div>

---

Every LLM forgets everything the moment a session ends.

`memex` fixes that — without cloud lock-in, telemetry, or a separate vector database process. Your memories live in a single SQLite file on your machine. You own them completely.

```python
from memex import Memory

mem = Memory()
mem.save("The user prefers concise answers and hates emojis")
mem.save("Working on a FastAPI backend with PostgreSQL")

# Ask it anything, semantically
print(mem.recall("what stack are we using?"))
# → "Working on a FastAPI backend with PostgreSQL"

# Inject context into any LLM call automatically
prompt = mem.inject("How should I name these API routes?")
# → "[Memory]\n- Working on a FastAPI backend...\n\nHow should I name..."
```

---

## Why memex

| | memex | mem0 | LangChain buffer | Roll your own |
|---|:---:|:---:|:---:|:---:|
| Fully local (no cloud required) | ✅ | ❌ | ✅ | ✅ |
| Semantic search | ✅ | ✅ | ❌ | ⚠️ |
| Cross-session persistence | ✅ | ✅ | ❌ | ⚠️ |
| Zero infrastructure | ✅ | ❌ | ✅ | ❌ |
| Works with any LLM | ✅ | ✅ | ✅ | ✅ |
| Auto-learns from conversations | ✅ | ✅ | ❌ | ❌ |
| Storage you can inspect/backup | ✅ | ❌ | ❌ | ⚠️ |
| `pip install` + done | ✅ | ❌ | ❌ | ❌ |

---

## Install

```bash
# Core (dependency-free, deterministic hash embedder)
pip install memex-ai

# + semantic embeddings via MiniLM (recommended)
pip install "memex-ai[local]"

# + REST server
pip install "memex-ai[server]"

# Node.js
npm install memex-ai
```

> **No GPU required.** MiniLM runs on CPU in ~5ms per call. The model (~23 MB) downloads once on first use.

---

## Quickstart

### Save and recall

```python
from memex import Memory

mem = Memory()  # persists to ~/.memex/default.db

mem.save("User's name is Layla")
mem.save("Layla is building a React app with a Node backend")
mem.save("She prefers TypeScript over JavaScript")

print(mem.recall("what language does the user prefer?"))
# → "She prefers TypeScript over JavaScript"
```

### Search top-k memories

```python
results = mem.search("tech stack", k=3)
for r in results:
    print(f"{r.score:.2f}  {r.text}")
# 0.94  Layla is building a React app with a Node backend
# 0.81  She prefers TypeScript over JavaScript
# 0.61  User's name is Layla
```

### Auto-learn from conversations

```python
# memex extracts durable facts from an exchange automatically
facts = mem.learn(
    user="My deadline is June 30th and I'm using Vercel for deploys.",
    assistant="Got it — I'll keep that in mind."
)
# facts → ["User's deadline is June 30th", "User deploys via Vercel"]
```

### Inject into any LLM

```python
# Prepend relevant memories to a user prompt
enriched = mem.inject("What framework should I use for auth?")

# Or get a ready-made system prompt string
system = mem.inject_system("What framework should I use for auth?")
```

---

## Integrations

### OpenAI

```python
from memex import Memory
from openai import OpenAI

mem = Memory()
client = OpenAI()

def chat(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": mem.inject_system(user_message)},
            {"role": "user",   "content": user_message},
        ]
    )
    reply = response.choices[0].message.content
    mem.learn(user=user_message, assistant=reply)   # learn facts for next time
    return reply
```

### Anthropic

```python
from memex import Memory
import anthropic

mem = Memory()
client = anthropic.Anthropic()

def chat(user_message: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=mem.inject_system(user_message),
        messages=[{"role": "user", "content": user_message}],
    )
    reply = response.content[0].text
    mem.learn(user=user_message, assistant=reply)
    return reply
```

### Ollama (100% local — no API keys, ever)

```python
from memex import Memory
import ollama

mem = Memory()

def chat(user_message: str) -> str:
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": mem.inject(user_message)}],
    )
    reply = response["message"]["content"]
    mem.learn(user=user_message, assistant=reply)
    return reply
```

### LangChain

```python
from memex.integrations.langchain import MemexMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# Drop-in replacement for ConversationBufferMemory
# Memories persist across sessions and are retrieved semantically
chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=MemexMemory(namespace="my-chatbot"),
)
```

---

## API reference

### `Memory(path?, namespace?, embedder?, auto_dedupe?, max_memories?)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `~/.memex/default.db` | Path to SQLite database |
| `namespace` | `str` | `"default"` | Isolates memories per user or app |
| `embedder` | `str \| Embedder` | `"auto"` | See embedder options below |
| `auto_dedupe` | `bool` | `True` | Merge near-duplicate memories |
| `max_memories` | `int` | `10_000` | Prunes oldest memories when exceeded |

### Methods

| Method | Returns | Description |
|---|---|---|
| `save(text, metadata?, importance?, ttl_days?)` | `str` | Store a memory, returns its ID |
| `save_many(texts)` | `list[str]` | Batch insert, faster than looping |
| `recall(query)` | `str \| None` | Best matching memory as plain text |
| `search(query, k=5, threshold=0.0, filters={})` | `list[MemoryRecord]` | Top-k memories with scores |
| `learn(user, assistant)` | `list[str]` | Extract and save facts from an exchange |
| `inject(prompt, k=5)` | `str` | Prepend memories to a user prompt |
| `inject_system(prompt, k=5)` | `str` | Return a system-prompt string with memories |
| `forget(query)` | `int` | Delete memories matching a query |
| `clear()` | `None` | Wipe all memories in the namespace |
| `export(path)` | `None` | Write memories to a JSON file |
| `import_from(path)` | `int` | Load memories from a JSON export |

### Embedders

```python
Memory(embedder="auto")       # MiniLM if installed, else hash (default)
Memory(embedder="hash")       # deterministic, dependency-free
Memory(embedder="minilm")     # all-MiniLM-L6-v2 — best speed/quality
Memory(embedder="bge-small")  # BAAI/bge-small-en-v1.5 — higher quality
Memory(embedder="nomic")      # nomic-embed-text-v1 — highest quality
Memory(embedder="openai")     # text-embedding-3-small (cloud, opt-in)
```

Custom embedder — implement two methods and a dimension:

```python
class MyEmbedder:
    dimension = 1536
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

mem = Memory(embedder=MyEmbedder())
```

### Namespaces

```python
# Memories are completely isolated between namespaces
alice = Memory(namespace="user:alice")
bob   = Memory(namespace="user:bob")

alice.save("Alice prefers metric units")
bob.search("units")  # → []  (bob sees nothing)
```

### TTL and structured metadata

```python
# Expire after N days
mem.save("User is currently in Dubai", ttl_days=7)

# Attach structured metadata for filtered search
mem.save("Deadline is June 30th", metadata={"category": "calendar"})
results = mem.search("deadlines", filters={"category": "calendar"})
```

---

## CLI

```bash
memex save "I use zsh with oh-my-zsh"
memex recall "what shell do I use?"
memex search "preferences" -k 5
memex list
memex forget "what shell do I use?"
memex clear
memex export ~/memories.json
memex import ~/memories.json
memex stats
# → 247 memories  |  12.3 MB  |  last updated 2 min ago
memex serve --port 8765
```

---

## REST server

For non-Python apps — Go, Rust, Ruby, plain HTTP.

```bash
pip install "memex-ai[server]"
memex serve --host 127.0.0.1 --port 8765
```

| Method | Endpoint | Body / query | Response |
|---|---|---|---|
| `POST` | `/save` | `{text, metadata?, ttl_days?}` | `{id}` |
| `GET` | `/recall` | `?q=...` | `{text}` |
| `GET` | `/search` | `?q=...&k=5` | `{results:[...]}` |
| `POST` | `/learn` | `{user_msg, assistant_msg}` | `{saved:[...]}` |
| `DELETE` | `/forget` | `?q=...` | `{deleted:N}` |
| `DELETE` | `/clear` | — | `{ok:true}` |
| `GET` | `/stats` | — | `{count, size_mb, ...}` |
| `GET` | `/export` | — | JSON download |
| `POST` | `/import` | JSON upload | `{imported:N}` |

```bash
curl -X POST http://127.0.0.1:8765/save \
  -H "content-type: application/json" \
  -d '{"text": "User prefers local tools"}'

curl "http://127.0.0.1:8765/recall?q=tool+preference"
# → {"text": "User prefers local tools"}
```

---

## JavaScript / TypeScript

The JS package mirrors the Python API exactly.

```typescript
import { Memory } from "memex-ai";

const mem = new Memory();
await mem.save("User clicked the dark mode toggle");

const result = await mem.recall("UI theme preference");
console.log(result); // → "User clicked the dark mode toggle"

const matches = await mem.search("preferences", { k: 5 });
```

Point it at a Python REST server for full semantic search and shared storage:

```typescript
const mem = new Memory({ endpoint: "http://127.0.0.1:8765" });
```

---

## How it works

```
your prompt
     │
     ▼ embed (MiniLM, ~5ms)
 [query vector]
     │
     ▼ KNN via sqlite-vec
 ┌───────────────────────────────┐
 │  ~/.memex/default.db          │
 │  ┌─────────────┐  ┌────────┐  │
 │  │  memories   │  │  vecs  │  │
 │  │  (text +    │◄─│ (ANN   │  │
 │  │   metadata) │  │ index) │  │
 │  └─────────────┘  └────────┘  │
 └───────────────────────────────┘
     │ top-k results
     ▼
 injected into your LLM call
```

**Storage**: Plain SQLite + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for approximate nearest-neighbor search. No daemon, no Docker, no port. Your memories are a single `.db` file — copy it, back it up, open it with any SQLite browser.

**Embeddings**: `all-MiniLM-L6-v2` (384 dims, ~23 MB, CPU-only, Apache 2.0). When `sentence-transformers` is not installed, memex falls back to a deterministic hash embedder so your app still works — just without semantic search.

**Deduplication**: On each `save()`, memex checks cosine similarity against existing memories. Anything above 0.92 is treated as a duplicate — the older entry is soft-deleted.

**Pruning**: When `max_memories` is reached, the lowest-importance memories are pruned asynchronously — `save()` never stalls.

---

## Benchmarks

Run locally:

```bash
python benchmarks/bench_memex.py --count 10000 --queries 100
```

Smoke result (Windows / Python 3.14 / sqlite-vec / hash embedder):

| corpus | `save` p50 | `search` p50 |
|---:|---:|---:|
| 10,000 | 0.3 ms | 15 ms |

Targets on the MiniLM path (Linux / sqlite-vec):

| operation | target |
|---|---:|
| `save()` | < 15 ms |
| `search()` | < 15 ms |
| import-time startup | < 300 ms |

> Windows tail latency may vary due to SQLite I/O behavior. Linux CI numbers are the reference. See [benchmarks/](benchmarks/) for methodology.

---

## Security

- Database files are created at `~/.memex/` with `0700` directory and `0600` file permissions on POSIX.
- All SQL uses parameter binding — no string interpolation, no injection surface.
- Metadata imports are JSON-validated and size-bounded before insertion.
- REST server binds to `127.0.0.1` by default; `--host 0.0.0.0` requires explicit opt-in.
- Zero telemetry. No network calls unless you configure a cloud embedder.

---

## FAQ

**Does memex ever call OpenAI or Anthropic?**  
No. The only network calls happen when you explicitly select a cloud embedder (`embedder="openai"`) or pass your LLM client to `learn()`. Core storage and retrieval are 100% local.

**Why is MiniLM opt-in?**  
`pip install memex-ai` shouldn't pull in PyTorch. The hash embedder works out of the box; `memex-ai[local]` adds the semantic layer when you're ready.

**How does memex compare to mem0?**  
mem0's hosted product sends your memories to their cloud. Their open-source version is still oriented around their managed service. memex has no cloud component — your `.db` file is the product.

**Can I use it with multiple users?**  
Yes — namespaces isolate memories inside one database file. Use `Memory(namespace=f"user:{user_id}")`.

**What happens when sqlite-vec isn't available?**  
memex falls back to a pure-Python cosine scan. Correct results at any corpus size; slower beyond ~5k memories. Install `memex-ai[local]` for the accelerated path.

**Is the storage format stable?**  
Yes. Schema migrations run automatically on open. Old `.db` files are always supported forward.

---

## Roadmap

- [x] SQLite + sqlite-vec backend
- [x] MiniLM / BGE / Nomic / OpenAI embedders
- [x] OpenAI, Anthropic, Ollama integrations
- [x] LangChain drop-in memory
- [x] Auto-learn from conversations
- [x] CLI + REST server
- [x] TypeScript package
- [ ] LlamaIndex integration
- [ ] Automatic memory summarization (compress old memories into abstractions)
- [ ] Browser extension (persist memory across ChatGPT, Claude.ai, Gemini)
- [ ] Multi-device sync — opt-in, end-to-end encrypted
- [ ] Rust core for 10× throughput (same Python/JS API)

---

## Contributing

Issues and PRs are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
git clone https://github.com/yourname/memex
cd memex
pip install -e ".[dev,local,server]"
pytest tests/
```

Good first issues are tagged [`good first issue`](https://github.com/yourname/memex/issues?q=label%3A%22good+first+issue%22). High-value areas: Gemini / Mistral / Cohere integrations, better deduplication heuristics, JS WASM embedder performance.

---

## License

MIT © 2026 — free for personal and commercial use.

---

<div align="center">
<sub>If memex saved you from building this yourself, a ⭐ goes a long way.</sub>
</div>