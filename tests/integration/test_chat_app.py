import json

from fastapi.testclient import TestClient

from memex.chat_app.app import create_chat_app
from memex.chat_app.llm import LocalDemoLLM
from memex.core import Memory


def test_chat_app_streams_reply_and_creates_visible_memories(tmp_path) -> None:
    memory = Memory(
        path=tmp_path / "chat.db",
        embedder="hash",
        use_sqlite_vec=False,
        auto_dedupe=False,
    )
    app = create_chat_app(memory=memory, llm=LocalDemoLLM(), namespace="demo")
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "Memex Chat" in page.text

    response = client.post(
        "/api/chat",
        json={
            "session_id": "test-session",
            "message": "My name is Riley. I prefer concise Python examples.",
        },
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    event_names = [event["event"] for event in events]
    assert "memory_context" in event_names
    assert "token" in event_names
    assert "memory_created" in event_names
    assert "done" in event_names

    memories = client.get("/api/memories").json()["memories"]
    memory_types = {memory["memory_type"] for memory in memories}
    assert {"short_term", "semantic", "episodic"}.issubset(memory_types)
