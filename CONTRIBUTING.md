# Contributing

Thanks for helping build memex.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy memex
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Design principles

- Keep the core local-first.
- Avoid network calls unless users explicitly opt in.
- Prefer SQLite and small local adapters over external infrastructure.
- Keep the public API small.
- Make heavy dependencies optional or lazy.
- Tests must not download models or require API keys.
- Preserve traceability when compressing memories.
- Keep sync transports optional, encrypted, and zero-trust.
- Add Rust acceleration behind a fallback path, never as a hard runtime requirement.

## Pull requests

Please include:

- A clear problem statement.
- Tests for changed behavior.
- Documentation updates for public API changes.
- Benchmark notes for storage, embedding, sync, or search changes.

## Release checklist

1. Update `CHANGELOG.md`.
2. Run `pytest`, `ruff`, `mypy`, and JS `npm test` when JS changed.
3. Build Python and npm packages.
4. Tag `vX.Y.Z`.
5. Publish from CI with trusted publishing.
