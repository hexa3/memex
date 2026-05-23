"""FastAPI chat web app for the Memex demo."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from memex.core import Memory
from memex.errors import OptionalDependencyError
from memex.models import MemoryKind, MemoryRecord
from memex.utils import default_data_dir, normalize_namespace

from .llm import ChatLLM, ChatMessage, create_llm

STATIC_DIR = Path(__file__).parent / "static"
CHAT_MEMORY_TYPES: list[MemoryKind] = ["short_term", "long_term", "episodic", "semantic"]


class ChatBody(BaseModel):
    """Request body for a streamed chat turn."""

    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=128)


class SessionState(BaseModel):
    """In-process chat history for one browser session."""

    messages: list[ChatMessage] = Field(default_factory=list)


def create_chat_app(
    *,
    memory: Memory | None = None,
    llm: ChatLLM | None = None,
    namespace: str = "chat-demo",
) -> Any:
    """Create the complete Memex chat demo web app."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise OptionalDependencyError(
            "The chat web app requires FastAPI. Install with: pip install 'memex-ai[server]'"
        ) from exc

    target_namespace = normalize_namespace(namespace)
    mem = memory or Memory(path=default_data_dir() / "chat-demo.db", namespace=target_namespace)
    chat_llm = llm or create_llm()
    sessions: dict[str, SessionState] = {}

    app = FastAPI(
        title="Memex Chat Demo",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str | bool]:
        return {"ok": True, "llm": chat_llm.provider}

    @app.get("/api/memories")
    def memories() -> dict[str, list[dict[str, Any]]]:
        records = mem.list(namespace=target_namespace, limit=500)
        return {"memories": [_memory_payload(record) for record in records]}

    @app.post("/api/chat")
    def chat(body: ChatBody) -> StreamingResponse:
        session_id = body.session_id or str(uuid.uuid4())
        state = sessions.setdefault(session_id, SessionState())
        user_message = body.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        return StreamingResponse(
            _chat_events(
                mem=mem,
                llm=chat_llm,
                state=state,
                session_id=session_id,
                user_message=user_message,
                namespace=target_namespace,
            ),
            media_type="application/x-ndjson",
        )

    return app


def serve_chat_app(
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    namespace: str = "chat-demo",
) -> None:
    """Run the chat demo app with uvicorn."""

    try:
        import uvicorn
    except ImportError as exc:
        raise OptionalDependencyError(
            "uvicorn is required for the chat web app. Install with: pip install 'memex-ai[server]'"
        ) from exc
    uvicorn.run(create_chat_app(namespace=namespace), host=host, port=port)


def _chat_events(
    *,
    mem: Memory,
    llm: ChatLLM,
    state: SessionState,
    session_id: str,
    user_message: str,
    namespace: str,
) -> Iterator[str]:
    memories = mem.hybrid_search(
        user_message,
        k=5,
        namespace=namespace,
        memory_types=CHAT_MEMORY_TYPES,
    )
    yield _event(
        "memory_context",
        {
            "session_id": session_id,
            "memories": [_memory_payload(memory) for memory in memories],
            "llm": llm.provider,
        },
    )

    short_term_id = mem.save(
        f"Recent message: {user_message}",
        namespace=namespace,
        memory_type="short_term",
        ttl_days=7,
        importance=0.25,
        metadata={"source": "chat", "session_id": session_id},
    )
    short_term = mem.storage.get(short_term_id)
    if short_term is not None:
        yield _event("memory_created", {"memory": _memory_payload(short_term)})

    assistant_chunks: list[str] = []
    try:
        for chunk in llm.stream(
            user_message=user_message,
            history=state.messages,
            memories=memories,
        ):
            assistant_chunks.append(chunk)
            yield _event("token", {"text": chunk})
    except Exception as exc:
        fallback = (
            "The configured LLM provider failed, so I am using the local demo response. "
            "Your memory was still retrieved and saved."
        )
        assistant_chunks = [fallback]
        yield _event("token", {"text": fallback})
        yield _event("warning", {"message": str(exc)})

    assistant_message = "".join(assistant_chunks).strip()
    created = _save_memories_from_turn(
        mem=mem,
        user_message=user_message,
        assistant_message=assistant_message,
        session_id=session_id,
        namespace=namespace,
    )
    for record in created:
        yield _event("memory_created", {"memory": _memory_payload(record)})

    state.messages.extend(
        [
            ChatMessage(role="user", content=user_message),
            ChatMessage(role="assistant", content=assistant_message),
        ]
    )
    state.messages = state.messages[-20:]
    yield _event("done", {"session_id": session_id, "message": assistant_message})


def _save_memories_from_turn(
    *,
    mem: Memory,
    user_message: str,
    assistant_message: str,
    session_id: str,
    namespace: str,
) -> list[MemoryRecord]:
    created: list[MemoryRecord] = []
    for text, memory_type in _extract_demo_memories(user_message):
        memory_id = mem.save(
            text,
            namespace=namespace,
            memory_type=memory_type,
            importance=0.9 if memory_type in {"long_term", "semantic"} else 0.55,
            metadata={"source": "chat_extraction", "session_id": session_id},
        )
        record = mem.storage.get(memory_id)
        if record is not None:
            created.append(record)

    episodic_id = mem.save(
        (
            f"Chat episode: user asked {user_message!r}; "
            f"assistant replied {assistant_message[:240]!r}."
        ),
        namespace=namespace,
        memory_type="episodic",
        importance=0.4,
        metadata={"source": "chat_episode", "session_id": session_id},
    )
    episodic = mem.storage.get(episodic_id)
    if episodic is not None:
        created.append(episodic)
    return created


def _extract_demo_memories(user_message: str) -> list[tuple[str, MemoryKind]]:
    text = user_message.strip()
    lowered = text.lower()
    memories: list[tuple[str, MemoryKind]] = []
    patterns: list[tuple[str, str, MemoryKind]] = [
        ("my name is ", "User's name is {value}.", "semantic"),
        ("i am ", "User is {value}.", "semantic"),
        ("i'm ", "User is {value}.", "semantic"),
        ("i live in ", "User lives in {value}.", "semantic"),
        ("i prefer ", "User prefers {value}.", "semantic"),
        ("i like ", "User likes {value}.", "semantic"),
        ("my goal is ", "User's goal is {value}.", "long_term"),
        ("remember that ", "{value}.", "long_term"),
    ]
    for marker, template, memory_type in patterns:
        index = lowered.find(marker)
        if index == -1:
            continue
        raw = text[index + len(marker) :]
        value = raw.split(".")[0].split("!")[0].split("?")[0].strip(" ,;:")
        if not value:
            continue
        formatted_value = value[0].upper() + value[1:] if marker == "remember that " else value
        formatted = template.format(value=formatted_value)
        item = (formatted, memory_type)
        if item not in memories:
            memories.append(item)
    return memories[:4]


def _memory_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "text": record.text,
        "memory_type": record.memory_type,
        "created_at": record.created_at.isoformat(),
        "score": record.score,
        "similarity": record.similarity,
        "metadata": record.metadata or {},
    }


def _event(
    event: Literal["memory_context", "memory_created", "token", "warning", "done"],
    data: Any,
) -> str:
    return json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n"
