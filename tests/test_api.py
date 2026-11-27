"""API: real model predictions, feedback logging into SQLite, and feedback changing the ranking."""
import pytest
from fastapi.testclient import TestClient

from tests.test_predict import fake_serving


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REVISE_MALIN_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("REVISE_MALIN_SERVING", str(fake_serving(tmp_path / "serving.npz")))
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_recommend_uses_the_model(client):
    r = client.get("/recommend", params={"student_id": 3, "k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "model" and len(body["recommendations"]) == 10
    scores = [x["score"] for x in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    first = body["recommendations"][0]
    assert first["score"] == pytest.approx(first["graph_score"] + first["time_score"], abs=1e-3)


def test_unknown_student_is_404(client):
    assert client.get("/recommend", params={"student_id": 999}).status_code == 404


def test_log_interaction_is_stored(client):
    r = client.post("/log_interaction", json={"student_id": 3, "topic": 5, "action": "revised"})
    assert r.status_code == 200 and r.json()["id"] == 1
    stored = client.get("/interactions/3").json()
    assert stored[0]["topic"] == 5 and stored[0]["action"] == "revised"


def test_invalid_action_is_rejected(client):
    r = client.post("/log_interaction", json={"student_id": 3, "topic": 5, "action": "liked"})
    assert r.status_code == 422


def test_hidden_topic_leaves_the_list(client):
    top = client.get("/recommend", params={"student_id": 3, "k": 5}).json()["recommendations"][0]["topic"]
    client.post("/log_interaction", json={"student_id": 3, "topic": top, "action": "already_known"})
    topics = [x["topic"] for x in client.get("/recommend", params={"student_id": 3, "k": 5}).json()["recommendations"]]
    assert top not in topics
    client.delete("/interactions/3")
    topics = [x["topic"] for x in client.get("/recommend", params={"student_id": 3, "k": 5}).json()["recommendations"]]
    assert topics[0] == top


def test_profile(client):
    p = client.get("/students/3").json()
    assert p["answers"] > 0 and len(p["by_level"]) == 3 and p["weakest_topics"]
