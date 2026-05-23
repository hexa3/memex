"""Optional FastAPI REST server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from memex.core import Memory
from memex.errors import OptionalDependencyError


class SaveBody(BaseModel):
    """Request body for ``POST /save``."""

    text: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] | None = None
    ttl_days: int | None = Field(default=None, ge=1)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    namespace: str | None = None


class LearnBody(BaseModel):
    """Request body for ``POST /learn``."""

    user_msg: str = Field(min_length=1, max_length=20_000)
    assistant_msg: str = Field(min_length=1, max_length=20_000)
    namespace: str | None = None
    metadata: dict[str, Any] | None = None


class ImportBody(BaseModel):
    """Request body for ``POST /import``."""

    payload: dict[str, Any]
    namespace: str | None = None
    replace: bool = False


def create_app(memory: Memory | None = None) -> Any:
    """Create a FastAPI application for a local memex server."""

    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise OptionalDependencyError(
            "FastAPI support is optional. Install with: pip install 'memex-ai[server]'"
        ) from exc

    mem = memory or Memory()
    app = FastAPI(
        title="memex",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/save")
    def save(body: SaveBody) -> dict[str, str]:
        try:
            memory_id = mem.save(
                body.text,
                metadata=body.metadata,
                ttl_days=body.ttl_days,
                importance=body.importance,
                namespace=body.namespace,
            )
            return {"id": memory_id}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/recall")
    def recall(
        q: str = Query(min_length=1),
        threshold: float = Query(default=0.0, ge=-1.0, le=1.0),
        namespace: str | None = None,
    ) -> dict[str, str | None]:
        return {"text": mem.recall(q, threshold=threshold, namespace=namespace)}

    @app.get("/search")
    def search(
        q: str = Query(min_length=1),
        k: int = Query(default=5, ge=1, le=100),
        threshold: float = Query(default=0.0, ge=-1.0, le=1.0),
        namespace: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        results = mem.search(q, k=k, threshold=threshold, namespace=namespace)
        return {
            "results": [
                result.model_dump(mode="json", exclude={"embedding"}, exclude_none=True)
                for result in results
            ]
        }

    @app.post("/learn")
    def learn(body: LearnBody) -> dict[str, list[str]]:
        try:
            facts = mem.learn(
                body.user_msg,
                body.assistant_msg,
                namespace=body.namespace,
                metadata=body.metadata,
            )
            return {"saved": facts}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/forget")
    def forget(
        q: str = Query(min_length=1),
        k: int = Query(default=5, ge=1, le=100),
        threshold: float = Query(default=0.75, ge=-1.0, le=1.0),
        namespace: str | None = None,
    ) -> dict[str, int]:
        return {"deleted": mem.forget(q, k=k, threshold=threshold, namespace=namespace)}

    @app.delete("/clear")
    def clear(namespace: str | None = None) -> dict[str, bool | int]:
        deleted = mem.clear(namespace=namespace)
        return {"ok": True, "deleted": deleted}

    @app.get("/stats")
    def stats(namespace: str | None = None) -> dict[str, Any]:
        return mem.stats(namespace=namespace).model_dump(mode="json")

    @app.get("/export")
    def export(namespace: str | None = None) -> JSONResponse:
        payload = mem.export(namespace=namespace)
        return JSONResponse(
            payload,
            headers={"Content-Disposition": 'attachment; filename="memex-export.json"'},
        )

    @app.post("/import")
    def import_records(body: ImportBody) -> dict[str, int]:
        try:
            imported = mem.import_from(
                body.payload,
                namespace=body.namespace,
                replace=body.replace,
            )
            return {"imported": imported}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str | Path | None = None,
    namespace: str = "default",
) -> None:
    """Run the local REST server."""

    try:
        import uvicorn
    except ImportError as exc:
        raise OptionalDependencyError(
            "uvicorn is required for serving. Install with: pip install 'memex-ai[server]'"
        ) from exc
    memory = Memory(path=path, namespace=namespace)
    uvicorn.run(create_app(memory), host=host, port=port)
