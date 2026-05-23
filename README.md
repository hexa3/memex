<div align="center">

<br/>

```
 ███╗   ███╗███████╗███╗   ███╗███████╗██╗  ██╗
 ████╗ ████║██╔════╝████╗ ████║██╔════╝╚██╗██╔╝
 ██╔████╔██║█████╗  ██╔████╔██║█████╗   ╚███╔╝ 
 ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝   ██╔██╗ 
 ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████╗██╔╝ ██╗
 ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
```

**Your AI finally remembers you.**

*Local-first. No cloud. No lock-in. One import.*

<br/>

[![PyPI](https://img.shields.io/pypi/v/memex-ai?style=flat-square&color=000000&label=pypi)](https://pypi.org/project/memex-ai/)
[![npm](https://img.shields.io/npm/v/memex-ai?style=flat-square&color=000000&label=npm)](https://www.npmjs.com/package/memex-ai)
[![Python](https://img.shields.io/badge/python-3.10%2B-000000?style=flat-square)](https://pypi.org/project/memex-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-000000?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/ibnshafi/memex/ci.yml?style=flat-square&color=000000&label=tests)](https://github.com/ibnshafi/memex/actions)
[![Stars](https://img.shields.io/github/stars/ibnshafi/memex?style=flat-square&color=000000)](https://github.com/ibnshafi/memex/stargazers)

<br/>

</div>

---

You ask your AI assistant the same question you asked last Tuesday.

It has no idea who you are.

Every session starts from zero. Every preference re-explained. Every project re-introduced. The context you spent months building — gone the moment the tab closes.

**memex ends that.**

It is a persistent, searchable memory layer that lives on your machine. A single SQLite file. No server to run. No API key to manage. No data leaving your device. Every LLM you use — OpenAI, Anthropic, Ollama, anything — gets a brain that survives across sessions, across projects, across years.

```python
from memex import Memory

mem = Memory()

mem.save("The user prefers concise Python examples.", memory_type="semantic")
mem.save("The project uses FastAPI and PostgreSQL.", memory_type="episodic")

print(mem.recall("what stack are we using?"))
# → "The project uses FastAPI and PostgreSQL."

prompt = mem.inject("How should I design the next endpoint?")
# → "[Memory]\n- The project uses FastAPI...\n\nHow should I design..."
```

---

## Why memex exists

| What you want | What you get today | What memex gives you |
|---|---|---|
| AI that remembers your name | Starts blank every session | Persistent cross-session memory |
| AI that knows your stack | You re-explain it every time | Semantic retrieval of past context |
| AI that learns your preferences | Forgets immediately | Long-term preference storage |
| Your data on your machine | Cloud lock-in | Local SQLite, zero telemetry |
| Works with any LLM | Vendor-specific memory | Provider-agnostic by design |
| Simple to add to any project | Complex integrations | One import, one file |

---

## The 30-second install

```bash
# Core — works immediately, no model download
pip install memex-ai

# + Semantic embeddings (recommended)
pip install "memex-ai[local]"

# + REST server, LlamaIndex, sync
pip install "memex-ai[local,server,llamaindex,sync]"

# Node.js
npm install memex-ai
```

---

## What it does

### Five memory types

memex doesn't flatten everything into one undifferentiated pile. Memories have types that reflect how humans actually think:

```python
mem.save("User's name is Layla",                        memory_type="semantic")
mem.save("Layla is building a fintech dashboard",       memory_type="episodic")
mem.save("She hates boilerplate and loves type hints",  memory_type="long_term")
mem.save("Currently debugging a JWT refresh bug",       memory_type="short_term")
```

| Type | What it stores |
|---|---|
| `short_term` | Current working context, temporary state |
| `long_term` | Durable preferences, rules, style choices |
| `episodic` | Events, sessions, project moments, decisions |
| `semantic` | Distilled facts about the user or domain |
| `summary` | Compressed memory with traceable source records |

### Hybrid search

Not just vectors. Not just keywords. Both.

```python
# Semantic vector search
results = mem.search("what stack are we using?", k=5)

# Hybrid: vector + keyword + score ranking
results = mem.hybrid_search("atlas-77 deployment", k=3)

for result in results:
    print(result.score, result.memory_type, result.text)
```

### Auto-learn from conversations

```python
# Extract and save durable facts from any exchange automatically
facts = mem.learn(
    user="My deadline is June 30th and we're deploying on Vercel.",
    assistant="Got it — I'll keep the deadline and deploy target in mind."
)
# → ["User's deadline is June 30th", "Project deploys on Vercel"]
```

### Inject into any LLM call

```python
# Prepend relevant memories to a user prompt
enriched_prompt = mem.inject("How should I name these API routes?")

# Get a ready-made system prompt string
system = mem.inject_system("How should I name these API routes?")
```

### Automatic summarization with traceability

Old memories don't just disappear — they compress.

```python
summary = mem.summarize(min_sources=8, max_sources=50)

# The summary knows exactly which memories it came from
print(summary.source_ids)
# → ["id-001", "id-002", "id-007", ...]
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
    mem.learn(user=user_message, assistant=reply)
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

### Ollama — 100% local, zero API keys

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

### LangChain — drop-in memory replacement

```python
from memex.integrations.langchain import MemexMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# Replace ConversationBufferMemory with one line
# Memories persist across sessions. Retrieval is semantic.
chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=MemexMemory(namespace="my-assistant"),
)
```

### LlamaIndex — memory as retrieval nodes

```python
from memex import Memory
from memex.integrations.llamaindex import MemexContext

mem = Memory()
ctx = MemexContext(memory=mem, k=5, hybrid=True)

nodes = ctx.retrieve("what user context matters?")
# Returns TextNode objects when LlamaIndex is installed
# Returns plain dicts otherwise — works either way
```

### TypeScript / Node.js

```typescript
import { Memory } from "memex-ai";

const mem = new Memory();

await mem.save("User prefers local-first tools.", {
  memoryType: "semantic",
});

const results = await mem.hybridSearch("local-first preference", { k: 3 });
console.log(results[0]?.text);
```

---

## Full API reference

### `Memory(path?, namespace?, embedder?, auto_dedupe?, max_memories?)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `~/.memex/default.db` | SQLite database path |
| `namespace` | `str` | `"default"` | Isolates memories per user or app |
| `embedder` | `str \| Embedder` | `"auto"` | Embedding backend |
| `auto_dedupe` | `bool` | `True` | Merge near-duplicate memories |
| `max_memories` | `int` | `10_000` | Prunes on overflow |

### Methods

| Method | Returns | Description |
|---|---|---|
| `save(text, memory_type?, metadata?, importance?, ttl_days?)` | `str` | Store a memory, returns its ID |
| `learn(user, assistant)` | `list[str]` | Extract and save facts from an exchange |
| `recall(query, threshold?)` | `str \| None` | Best matching memory as plain text |
| `search(query, k?, threshold?, filters?)` | `list[MemoryRecord]` | Top-k semantic search |
| `hybrid_search(query, k?)` | `list[MemoryRecord]` | Vector + keyword combined search |
| `inject(prompt, k?)` | `str` | Prepend memories to a user prompt |
| `inject_system(prompt, k?)` | `str` | Ready-made system prompt with memories |
| `summarize(min_sources?, max_sources?)` | `SummaryRecord` | Compress old memories into a traceable summary |
| `forget(query, k?, threshold?)` | `int` | Delete memories matching a query |
| `clear()` | `int` | Wipe all memories in the namespace |
| `export(path?)` | `dict` | Export to JSON file or return payload |
| `import_from(path_or_payload, replace?)` | `int` | Import from JSON export |
| `start_cleanup_scheduler()` | `None` | Start background maintenance |
| `stop_cleanup_scheduler()` | `None` | Stop background maintenance |

### Embedders

```python
Memory(embedder="auto")       # MiniLM if installed, else hash (default)
Memory(embedder="hash")       # deterministic, zero dependencies
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

---

## CLI

```bash
memex save "I use zsh with oh-my-zsh"              --memory-type semantic
memex recall "what shell do I use?"
memex search "shell preferences" -k 5
memex hybrid-search "deployment setup"
memex list
memex forget "old job"
memex clear
memex export ~/memories.json
memex import ~/memories.json
memex stats
memex serve --port 8765
memex models
```

---

## REST server

```bash
pip install "memex-ai[server]"
memex serve --host 127.0.0.1 --port 8765
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/save` | Save a memory |
| `GET` | `/recall` | Best matching memory |
| `GET` | `/search` | Top-k semantic search |
| `GET` | `/hybrid-search` | Vector + keyword search |
| `POST` | `/learn` | Extract facts from exchange |
| `POST` | `/summarize` | Compress old memories |
| `DELETE` | `/forget` | Delete matching memories |
| `DELETE` | `/clear` | Wipe namespace |
| `GET` | `/stats` | Storage statistics |
| `GET` | `/export` | JSON download |
| `POST` | `/import` | JSON upload |

```bash
curl -X POST http://127.0.0.1:8765/save \
  -H "content-type: application/json" \
  -d '{"text": "User prefers local tools", "memory_type": "semantic"}'
```

---

## Browser extension

Capture memories from ChatGPT, Claude, and Gemini with a single click. No paste. No copy. Just hit **Save memory** and it's stored.

Supported sites: `chatgpt.com` · `claude.ai` · `gemini.google.com`

**Install:**

1. Open Chrome or Edge → Extensions → Enable developer mode
2. Click **Load unpacked**
3. Select the `extension/` folder in this repository
4. Open any supported AI site → click **Save memory**

**Privacy:** captures only on user action, deduplicates by content hash, stores locally, never sends passphrases anywhere, uses minimum permissions.

---

## Optional encrypted sync

Sync is off by default. When you enable it, everything is encrypted on-device before it leaves.

```bash
pip install "memex-ai[sync]"
```

```python
from memex.sync import CRDTState, EncryptedSyncCodec

codec = EncryptedSyncCodec.from_passphrase("your-long-private-passphrase")
envelope = codec.encrypt(b'{"records":[]}')
payload   = codec.decrypt(envelope)
```

**Security model:** ChaCha20-Poly1305 authenticated encryption. CRDT-style conflict resolution. Remote storage is treated as fully untrusted — it only ever sees ciphertext. Sync transport is intentionally separate so you can use your own backend.

---

## How it works

```
Your prompt
    │
    ▼  embed (MiniLM, ~5ms)
[query vector]
    │
    ├─── sqlite-vec KNN ──────────────────────┐
    │                                         │
    └─── keyword scan (hybrid fallback) ──────┤
                                              │
                                    top-k results
                                              │
                                              ▼
                              ┌───────────────────────────┐
                              │   ~/.memex/default.db     │
                              │   ┌──────────┐ ┌───────┐  │
                              │   │ memories │ │ vecs  │  │
                              │   │ (text +  │ │ (ANN) │  │
                              │   │ metadata)│ │       │  │
                              │   └──────────┘ └───────┘  │
                              └───────────────────────────┘
                                              │
                                    injected into LLM call
```

**Storage:** Plain SQLite + `sqlite-vec` for ANN search. No daemon. No Docker. No port. The database is a single file you can copy, inspect, or delete.

**Embeddings:** `all-MiniLM-L6-v2` by default — 384 dims, ~23MB, runs on CPU in ~5ms. Swappable with zero code changes.

**Summarization:** When memories accumulate, a background job compresses them into a `summary` record that stores the IDs of every source it absorbed. Nothing is lost, it's just compressed.

**Deduplication:** On every save, cosine similarity is checked against existing memories. Anything above 0.92 similarity is treated as a duplicate and the older entry is soft-deleted.

**Pruning:** When `max_memories` is hit, the lowest-importance memories are pruned asynchronously. `save()` never stalls.

---

## Optional Rust core

For high-throughput use cases, a Rust crate provides accelerated routines for cosine ranking, extractive summarization, and sync diff computation. Python falls back gracefully when Rust is not installed — the API is identical either way.

```bash
cd rust-core && cargo test && cargo bench
```

---

## Benchmarks

| corpus | operation | p50 |
|---:|---|---:|
| 10,000 | `save()` | 0.3 ms |
| 10,000 | `search()` | 15 ms |
| — | import-time startup | < 300 ms |

Run locally:

```bash
python benchmarks/bench_memex.py --count 10000 --queries 100
```

---

## Security

- Zero telemetry. No network calls in default configuration.
- Database created at `~/.memex/` with `0700` directory and `0600` file permissions on POSIX.
- All SQL uses parameter binding — no string interpolation, no injection surface.
- REST server binds to `127.0.0.1` by default. `--host 0.0.0.0` requires explicit opt-in.
- Browser extension stores captures locally, never stores passphrases, uses minimum permissions.
- Sync payloads are encrypted on-device before transport. Remote storage sees only ciphertext.

---

## FAQ

**Does memex ever call OpenAI or Anthropic without me knowing?**
No. The only network calls that happen are: model download on first use of MiniLM (one-time, cached), embedding calls if you explicitly select the OpenAI embedder, and whatever LLM calls your own code makes.

**Why is MiniLM opt-in?**
`pip install memex-ai` shouldn't silently pull in PyTorch. The hash embedder works immediately with no download. `memex-ai[local]` adds the semantic layer when you want it.

**How does memex compare to mem0?**
mem0's default configuration sends your memories to their cloud. memex has no cloud component. Your `.db` file is the product — copy it to a USB drive if you want.

**Can I use it with multiple users?**
Yes. `Memory(namespace=f"user:{user_id}")` completely isolates memories per user inside a single database file.

**Is the storage format stable?**
Yes. Schema migrations run automatically on connection open. Old databases are always forward-compatible.

**What happens when sqlite-vec isn't available?**
memex falls back to a pure-Python cosine similarity scan. Correct results at any scale. Slower at 10k+ memories. Install `memex-ai[local]` for the fast path.

**Can I self-host the sync backend?**
Yes. The sync layer provides encrypted envelopes and CRDT merge primitives. The transport is intentionally left separate so you can point it at S3, a self-hosted server, Dropbox, or anything else.

---

## Troubleshooting

**`memex` command not found**
```bash
python -m memex.cli --help
```
Or add your Python scripts directory to PATH.

**Search returns weak results**
```bash
pip install "memex-ai[local]"
```
The default hash embedder is offline-friendly but not semantic. MiniLM fixes this.

**FastAPI server won't start**
```bash
pip install "memex-ai[server]"
```

**Sync import fails**
```bash
pip install "memex-ai[sync]"
```

**Browser extension Save button missing**
Check the extension is loaded from `extension/`, the current site is supported, and the page has fully loaded.

**Browser capture says already saved**
memex deduplicates by content hash. Open a different conversation or reload before saving again.

---

## Roadmap

- [x] SQLite + sqlite-vec storage
- [x] Five memory types with decay and pruning
- [x] MiniLM / BGE / Nomic / OpenAI embedders
- [x] Hybrid vector + keyword search
- [x] Auto-learn from conversations
- [x] Automatic summarization with source tracing
- [x] OpenAI, Anthropic, Ollama integrations
- [x] LangChain drop-in memory
- [x] LlamaIndex adapter
- [x] CLI + REST server
- [x] TypeScript package
- [x] Browser extension (ChatGPT, Claude, Gemini)
- [x] Optional encrypted sync
- [x] Rust performance core
- [ ] `memex demo` — one-command chat UI with live memory sidebar
- [ ] VS Code extension — codebase memory for AI coding assistants
- [ ] Obsidian plugin — notes as memory, memory as context
- [ ] Memory visualization — 2D UMAP projection of your memory space
- [ ] Multi-device sync via S3 / self-hosted backend

---

## Contributing

Issues and PRs are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
git clone https://github.com/ibnshafi/memex
cd memex
pip install -e ".[dev,local,server]"
pytest tests/
ruff check .
mypy memex
```

```bash
cd js && npm install && npm run typecheck && npm test
cd rust-core && cargo test
```

Good first issues are tagged [`good first issue`](https://github.com/ibnshafi/memex/issues?q=label%3A%22good+first+issue%22). High-value areas: Gemini / Mistral / Cohere integrations, VS Code extension, Obsidian plugin, memory visualization.

---

## License

MIT © 2025 ibnshafi — free for personal and commercial use.

---

<div align="center">

<br/>

**memex is what AI memory should have been from the start.**

*Local. Private. Permanent. Yours.*

<br/>

If memex saved you from re-explaining yourself one more time — a ⭐ means everything.

<br/>

[⭐ Star](https://github.com/ibnshafi/memex) · [🍴 Fork](https://github.com/ibnshafi/memex/fork) · [🐛 Issues](https://github.com/ibnshafi/memex/issues) · [💬 Discussions](https://github.com/ibnshafi/memex/discussions)

<br/>

</div>
