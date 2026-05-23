# memex

Local-first memory for AI applications.

`memex` gives an assistant durable memory that can be saved, searched, summarized, exported, and reused across sessions. It runs locally by default, stores data in SQLite, supports Python and TypeScript, and keeps sync optional and encrypted.

## Why memex

Most AI apps forget everything when a chat ends. memex adds a small, inspectable memory layer:

- Save facts, preferences, project context, and conversation moments.
- Retrieve relevant memory with semantic search plus keyword fallback.
- Compress older memories into traceable summaries.
- Capture memories from ChatGPT, Claude, and Gemini with a local-first browser extension.
- Add optional encrypted sync without trusting a server with plaintext data.

## Key Features

- **Local-first storage**: SQLite database on the user's machine.
- **Hybrid retrieval**: vector similarity, keyword fallback, and score ranking.
- **Memory hierarchy**: `short_term`, `long_term`, `episodic`, `semantic`, and `summary`.
- **Automatic summarization**: compresses older memories while preserving `source_ids`.
- **LlamaIndex integration**: retrieve memex context as LlamaIndex-friendly nodes.
- **Browser extension**: clean capture overlay for ChatGPT, Claude, and Gemini.
- **Optional encrypted sync**: CRDT merge primitives and ChaCha20-Poly1305 envelopes.
- **Rust performance layer**: optional core routines for ranking, summarization, and sync diffing.
- **Python and TypeScript SDKs**: same local-first model from backend or frontend code.
- **REST server**: optional FastAPI server for local apps and tools.

## Architecture

```text
ChatGPT / Claude / Gemini     Python Apps        TypeScript Apps        CLI
          |                       |                    |                 |
          v                       v                    v                 v
   Browser Extension        Python SDK           TypeScript SDK      memex CLI
          |                       |                    |                 |
          +-----------------------+--------------------+-----------------+
                                  |
                                  v
                            Memory API
              save | learn | search | hybrid_search | summarize
                                  |
                +-----------------+------------------+
                |                                    |
                v                                    v
        SQLite Storage                         LlamaIndex Adapter
        sqlite-vec when available
        Python scan fallback
                |
                v
       Optional Rust Core Engine
                |
                v
 Optional Encrypted Sync Primitives
       disabled by default
```

## Installation

Python:

```bash
pip install memex-ai
```

TypeScript:

```bash
npm install memex-ai
```

Optional Python extras:

```bash
pip install "memex-ai[local,server,sync,llamaindex]"
```

## Quickstart

Copy, paste, and run:

```python
from memex import Memory

mem = Memory()

mem.save("The user prefers concise Python examples.", memory_type="semantic")
mem.save("The project uses FastAPI and PostgreSQL.", memory_type="episodic")

print(mem.recall("what stack are we using?"))

prompt = mem.inject("How should I design the next endpoint?")
print(prompt)
```

CLI:

```bash
memex save "The user prefers concise Python examples"
memex search "Python examples"
```

## API Examples

### Save Memory

```python
memory_id = mem.save(
    "The deployment codename is atlas-77.",
    memory_type="episodic",
    metadata={"source": "release-notes"},
)
```

### Hybrid Search

```python
results = mem.hybrid_search("atlas-77 deployment", k=3)

for result in results:
    print(result.score, result.memory_type, result.text)
```

### Summarize With Traceability

```python
summary = mem.summarize(min_sources=8, max_sources=50)

print(summary.summary_id)
print(summary.source_ids)
```

### Export And Import

```python
payload = mem.export("memex-export.json")
imported = mem.import_from(payload)
```

### TypeScript

```ts
import { Memory } from "memex-ai";

const mem = new Memory();

await mem.save("User prefers local-first tools.", {
  memoryType: "semantic",
});

const results = await mem.hybridSearch("local-first preference", { k: 3 });
console.log(results[0]?.text);
```

### REST Server

```bash
pip install "memex-ai[server]"
memex serve
```

Then call:

```bash
curl "http://127.0.0.1:8765/search?q=Python%20examples"
```

The server binds to `127.0.0.1` by default.

## Browser Extension Setup

The extension captures conversation text from supported AI sites and stores it in browser-local storage. Sync is off by default.

Supported sites:

- ChatGPT: `chatgpt.com`, `chat.openai.com`
- Claude: `claude.ai`
- Gemini: `gemini.google.com`

Install locally:

1. Open Chrome or Edge extension settings.
2. Enable developer mode.
3. Choose **Load unpacked**.
4. Select the `extension/` folder in this repository.
5. Open ChatGPT, Claude, or Gemini and use the **Save memory** button.

Firefox support uses the same WebExtension structure, but the current manifest is optimized for Chromium Manifest V3.

Extension behavior:

- Captures only when the user clicks **Save memory**.
- Deduplicates the current conversation view before saving.
- Keeps captures local when offline.
- Uses the minimum permission set: `storage` plus supported host matches.
- Does not store API tokens or passphrases.

## Optional Sync Setup

Sync is optional and disabled by default. The core package provides encrypted envelopes and CRDT merge logic; transport is intentionally separate so teams can use their own trusted backend.

Install sync support:

```bash
pip install "memex-ai[sync]"
```

Use the encrypted codec:

```python
from memex.sync import CRDTState, EncryptedSyncCodec

codec = EncryptedSyncCodec.from_passphrase("use a long private passphrase")
envelope = codec.encrypt(b'{"records":[]}')
payload = codec.decrypt(envelope)
```

Security model:

- No plaintext sync payloads should leave the device.
- Remote sync storage must be treated as untrusted.
- Conflicts are resolved with CRDT-style merge semantics.
- Browser extension sync remains off until the user enables it and provides an endpoint.

## Rust Core

The `rust-core/` crate contains optional performance routines for:

- cosine similarity and top-k ranking
- extractive summary helpers
- sync diff computation

Python uses a safe fallback when the Rust module is not installed. Maintainers can build Rust bindings with the crate feature flags:

```bash
cd rust-core
cargo test
cargo bench
```

The package remains usable without Rust.

## LlamaIndex Integration

Use `MemexContext` to retrieve memory as LlamaIndex-friendly nodes:

```python
from memex import Memory
from memex.integrations.llamaindex import MemexContext

mem = Memory()
ctx = MemexContext(memory=mem, k=5, hybrid=True)

nodes = ctx.retrieve("what user context matters?")
```

When LlamaIndex is installed, `as_llamaindex_nodes()` returns `TextNode` objects. Without LlamaIndex, it returns plain dictionaries.

## Memory System

memex stores both raw and compressed memory.

| Type | Use |
| --- | --- |
| `short_term` | Temporary context or recent working state |
| `long_term` | Durable facts and preferences |
| `episodic` | Events, conversations, sessions, and project moments |
| `semantic` | Distilled knowledge about the user or domain |
| `summary` | Compressed memory with traceable source records |

Summaries preserve traceability:

```python
summary = mem.summarize(min_sources=8)
record = mem.storage.get(summary.summary_id)
print(record.source_ids)
```

Background maintenance:

```python
mem.start_cleanup_scheduler()
mem.stop_cleanup_scheduler()
```

## Security And Encryption

Default posture:

- Local-first by default.
- No telemetry.
- No required cloud service.
- REST server binds to localhost unless configured otherwise.
- Browser extension stores captures locally and never stores passphrases.

Sync posture:

- Sync is disabled by default.
- Encrypted sync uses authenticated encryption through the `sync` extra.
- Remote sync transports are considered untrusted.
- Conflict handling is offline-first.

## Troubleshooting

### `memex` command not found

Make sure your Python scripts directory is on PATH, or run:

```bash
python -m memex.cli --help
```

### Search returns weak results

Install local semantic embeddings:

```bash
pip install "memex-ai[local]"
```

The default fallback embedder is deterministic and offline-friendly, but a local model improves semantic recall.

### FastAPI server will not start

Install the server extra:

```bash
pip install "memex-ai[server]"
```

### Sync encryption import fails

Install the sync extra:

```bash
pip install "memex-ai[sync]"
```

### Browser extension button is missing

Check that:

- The extension is loaded from the `extension/` folder.
- The current site is ChatGPT, Claude, or Gemini.
- The page has finished loading.
- The extension has host access for the site.

### Browser capture says the view is already saved

memex deduplicates the current conversation view by content hash. Change the conversation or reload a different chat before saving again.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy memex
```

JavaScript:

```bash
cd js
npm install
npm run typecheck
npm run build
npm test
```

Rust:

```bash
cd rust-core
cargo test
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API_REFERENCE.md](API_REFERENCE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
