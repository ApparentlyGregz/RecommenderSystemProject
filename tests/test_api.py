"""API: mock recommendations and real logging into SQLite."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REVISE_MALIN_DB", str(tmp_path / "test.db"))
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_recommend_returns_mock_list(client):
    r = client.get("/recommend", params={"student_id": 7, "k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "mock" and len(body["recommendations"]) == 10
    risks = [x["risk"] for x in body["recommendations"]]
    assert risks == sorted(risks, reverse=True)


def test_log_interaction_is_stored(client):
    r = client.post("/log_interaction", json={"student_id": 7, "topic": 52, "action": "revised"})
    assert r.status_code == 200 and r.json()["id"] == 1
    stored = client.get("/interactions/7").json()
    assert stored[0]["topic"] == 52 and stored[0]["action"] == "revised"


def test_invalid_action_is_rejected(client):
    r = client.post("/log_interaction", json={"student_id": 7, "topic": 52, "action": "liked"})
    assert r.status_code == 422
