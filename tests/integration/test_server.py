import pytest

from memex import Memory

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memex.server import create_app  # noqa: E402


def test_rest_save_and_search(tmp_path) -> None:
    mem = Memory(path=tmp_path / "memex.db", embedder="hash", use_sqlite_vec=False)
    client = TestClient(create_app(mem))

    response = client.post("/save", json={"text": "User prefers SQLite"})
    assert response.status_code == 200

    search = client.get("/search", params={"q": "SQLite", "k": 1})
    assert search.status_code == 200
    assert search.json()["results"][0]["text"] == "User prefers SQLite"
