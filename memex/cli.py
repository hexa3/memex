"""Command line interface for memex."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer

from memex import __version__
from memex.core import Memory
from memex.embedders import create_embedder
from memex.models import MemoryKind

app = typer.Typer(help="Local-first memory for LLM applications.", no_args_is_help=True)


def _memory(
    *,
    db: Path | None,
    namespace: str,
    embedder: str,
) -> Memory:
    return Memory(path=db, namespace=namespace, embedder=create_embedder(embedder))


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    """memex command line entrypoint."""

    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def save(
    text: str = typer.Argument(..., help="Memory text to save."),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db", help="SQLite database path."),
    embedder: str = typer.Option("auto", "--embedder", help="Embedder name."),
    ttl_days: int | None = typer.Option(None, "--ttl-days", min=1),
    importance: float = typer.Option(1.0, "--importance", min=0.0, max=1.0),
    memory_type: str = typer.Option("long_term", "--type", help="Memory hierarchy type."),
) -> None:
    """Save a memory."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    typer.echo(
        mem.save(
            text,
            ttl_days=ttl_days,
            importance=importance,
            memory_type=cast(MemoryKind, memory_type),
        )
    )


@app.command()
def recall(
    query: str = typer.Argument(...),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
    threshold: float = typer.Option(0.0, "--threshold", min=-1.0, max=1.0),
) -> None:
    """Return the best matching memory."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    result = mem.recall(query, threshold=threshold)
    if result is not None:
        typer.echo(result)


@app.command()
def search(
    query: str = typer.Argument(...),
    k: int = typer.Option(5, "-k", min=1, max=100),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
    threshold: float = typer.Option(0.0, "--threshold", min=-1.0, max=1.0),
    hybrid: bool = typer.Option(
        True,
        "--hybrid/--semantic-only",
        help="Blend vector and keyword ranking.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print JSON results."),
) -> None:
    """Search memories."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    if hybrid:
        results = mem.hybrid_search(query, k=k, threshold=threshold)
    else:
        results = mem.search(query, k=k, threshold=threshold)
    if json_output:
        typer.echo(
            json.dumps(
                [record.model_dump(mode="json", exclude_none=True) for record in results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for record in results:
        score = f"{record.score:.3f}" if record.score is not None else "n/a"
        typer.echo(f"{score}  {record.id}  {record.text}")


@app.command("list")
def list_memories(
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
    limit: int = typer.Option(100, "--limit", min=1, max=1000),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List memories."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    records = mem.list(limit=limit)
    if json_output:
        typer.echo(
            json.dumps(
                [record.model_dump(mode="json", exclude_none=True) for record in records],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for record in records:
        typer.echo(f"{record.id}  {record.created_at.isoformat()}  {record.text}")


@app.command()
def forget(
    query: str = typer.Argument(...),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
    k: int = typer.Option(5, "-k", min=1, max=100),
    threshold: float = typer.Option(0.75, "--threshold", min=-1.0, max=1.0),
) -> None:
    """Forget memories matching a query."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    typer.echo(mem.forget(query, k=k, threshold=threshold))


@app.command()
def clear(
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clear all memories in a namespace."""

    if not yes and not typer.confirm(f"Clear namespace '{namespace}'?"):
        raise typer.Exit(code=1)
    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    typer.echo(mem.clear())


@app.command("export")
def export_cmd(
    path: Path | None = typer.Argument(None),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
) -> None:
    """Export memories to JSON."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    payload = mem.export(path)
    if path is None:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(str(path))


@app.command("import")
def import_cmd(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
    replace: bool = typer.Option(False, "--replace"),
) -> None:
    """Import memories from JSON."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    typer.echo(mem.import_from(path, namespace=namespace, replace=replace))


@app.command()
def stats(
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
) -> None:
    """Show memory statistics."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    typer.echo(json.dumps(mem.stats().model_dump(mode="json"), indent=2))


@app.command()
def summarize(
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
    min_sources: int = typer.Option(8, "--min-sources", min=2),
    max_sources: int = typer.Option(50, "--max-sources", min=2, max=500),
    delete_sources: bool = typer.Option(
        False,
        "--delete-sources",
        help="Replace sources with summary.",
    ),
) -> None:
    """Create a traceable semantic summary from existing memories."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    result = mem.summarize(
        namespace=namespace,
        min_sources=min_sources,
        max_sources=max_sources,
        delete_sources=delete_sources,
    )
    typer.echo(json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2))


@app.command()
def optimize(
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
    embedder: str = typer.Option("auto", "--embedder"),
) -> None:
    """Run one memory cleanup and summarization pass."""

    mem = _memory(db=db, namespace=namespace, embedder=embedder)
    result = mem.optimize(namespace=namespace)
    typer.echo(json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port", min=1, max=65535),
    namespace: str = typer.Option("default", "--namespace", "-n"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """Start the local REST server."""

    if host == "0.0.0.0":
        typer.echo("Warning: binding to 0.0.0.0 exposes memex to your network.", err=True)
    from memex.server import serve as run_server

    run_server(host=host, port=port, path=db, namespace=namespace)


@app.command()
def chat(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8766, "--port", min=1, max=65535),
    namespace: str = typer.Option("chat-demo", "--namespace", "-n"),
) -> None:
    """Start the Memex chat web app."""

    if host == "0.0.0.0":
        typer.echo("Warning: binding to 0.0.0.0 exposes the chat demo to your network.", err=True)
    from memex.chat_app.app import serve_chat_app

    typer.echo(f"Memex Chat is starting at http://{host}:{port}")
    serve_chat_app(host=host, port=port, namespace=namespace)


@app.command()
def models() -> None:
    """List built-in embedders."""

    for name in ("auto", "hash", "minilm", "bge-small", "nomic", "openai"):
        typer.echo(name)


if __name__ == "__main__":
    app()
