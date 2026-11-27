"""API: real recommendations from a (tiny, fake) model bundle, questions, answers and feedback in SQLite."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

N_TOPICS = 4


@pytest.fixture
def bundle(tmp_path):
    """A 3-student, 4-skill bundle in the format written by src.export, plus topic and question tables."""
    rng = np.random.default_rng(0)
    last_ts = np.full((3, N_TOPICS), np.nan)
    last_ts[0, :3] = [1_000, 2_000, 3_000]
    np.savez_compressed(
        tmp_path / "serving.npz", user_ids=np.array([10, 11, 12]), user_emb=rng.normal(size=(3, 4)).astype(np.float32),
        topic_emb=rng.normal(size=(N_TOPICS, 4)).astype(np.float32), topic_bias=np.zeros(N_TOPICS, np.float32),
        half_life=np.full(N_TOPICS, 5.0, np.float32), beta=np.float32(-0.1), unseen=np.float32(0.2),
        use_forgetting=np.bool_(True), attempts=np.array([[4, 2, 1, 0], [1, 1, 1, 1], [0, 0, 3, 3]], np.int16),
        fails=np.array([[3, 0, 1, 0], [0, 1, 0, 1], [0, 0, 2, 0]], np.int16), last_ts=last_ts,
        now=np.array([4_000, 4_000, 4_000]), neighbours=np.array([[1, 2], [0, 2], [0, 1]], np.int16),
        questions=np.array([1, 2, 3], np.int32), answered=np.array([[True, False, False]] * 3),
        meta=np.array(json.dumps({"model": "full", "trained_epoch": 3, "config": {}})),
    )
    pd.DataFrame({"topic": range(N_TOPICS), "kc_id": [5, 6, 7, 8], "name": ["Join", "Keys", "SQL", "Locks"],
                  "description": ["d"] * N_TOPICS, "n_events": [9] * N_TOPICS, "n_questions": [2, 1, 1, 1]}
                 ).to_parquet(tmp_path / "topics.parquet")
    choices = [{"id": 100 + i, "text": f"choice {i}", "is_correct": i == 1} for i in range(3)]
    pd.DataFrame({
        "question": [1, 2, 3], "title": ["Q1", "Q2", "Q3"],
        "blocks": [json.dumps([{"type": "html", "content": "What is a join?"}])] * 3,
        "text": ["What is a join?"] * 3, "hint": ["", "think", ""], "explanation": ["Because.", "", ""],
        "difficulty": [1, 2, 3], "choices": [json.dumps(choices)] * 3,
        "topics": [json.dumps([0]), json.dumps([0, 1]), json.dumps([2])],
    }).to_parquet(tmp_path / "questions.parquet")
    return tmp_path


@pytest.fixture
def client(bundle, monkeypatch):
    monkeypatch.setenv("REVISE_MALIN_DB", str(bundle / "test.db"))
    monkeypatch.setenv("REVISE_MALIN_SERVING", str(bundle / "serving.npz"))
    monkeypatch.setenv("REVISE_MALIN_TOPICS", str(bundle / "topics.parquet"))
    monkeypatch.setenv("REVISE_MALIN_QUESTIONS", str(bundle / "questions.parquet"))
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_recommend_uses_the_model(client):
    body = client.get("/recommend", params={"student_id": 10, "k": 3}).json()
    assert body["mode"] == "model" and len(body["recommendations"]) == 3
    risks = [r["risk"] for r in body["recommendations"]]
    assert risks == sorted(risks, reverse=True)
    first = body["recommendations"][0]
    assert first["skill"] in {"Join", "Keys", "SQL", "Locks"} and first["reasons"]


def test_unknown_student_is_404(client):
    assert client.get("/recommend", params={"student_id": 999}).status_code == 404


def test_questions_hide_the_answer_and_prefer_unseen(client):
    qs = client.get("/questions", params={"student_id": 10, "topic": 0, "n": 2}).json()
    assert [q["question"] for q in qs] == [2, 1]  # question 1 was already answered in the dataset
    assert all("is_correct" not in c for q in qs for c in q["choices"])


def test_answer_is_checked_stored_and_changes_the_ranking(client):
    before = client.get("/recommend", params={"student_id": 11, "k": 4}).json()
    r = client.post("/answer", json={"student_id": 11, "question": 3, "choice": 101}).json()
    assert r["correct"] is True and r["correct_choices"] == [101]
    wrong = client.post("/answer", json={"student_id": 11, "question": 3, "choice": 100}).json()
    assert wrong["correct"] is False
    after = client.get("/recommend", params={"student_id": 11, "k": 4}).json()
    assert after["session_answers"] == 2 and before["session_answers"] == 0
    sql = next(x for x in after["recommendations"] if x["skill"] == "SQL")
    assert "In this session: 1 of 2 correct." in sql["reasons"]
    assert len(client.get("/interactions/11").json()["answers"]) == 2


def _risk(client, student, skill):
    recs = client.get("/recommend", params={"student_id": student, "k": 4}).json()["recommendations"]
    return next(r["risk"] for r in recs if r["skill"] == skill)


def test_wrong_answer_raises_priority_and_correct_lowers_it(client):
    before = _risk(client, 10, "SQL")
    client.post("/answer", json={"student_id": 10, "question": 3, "choice": 100})  # wrong
    after_wrong = _risk(client, 10, "SQL")
    assert after_wrong > before
    client.post("/reset/10")
    client.post("/answer", json={"student_id": 10, "question": 3, "choice": 101})  # correct
    assert _risk(client, 10, "SQL") < before


def test_choice_of_another_question_is_rejected(client):
    assert client.post("/answer", json={"student_id": 10, "question": 1, "choice": 999}).status_code == 422


def test_already_known_hides_the_skill_and_reset_restores_it(client):
    client.post("/log_interaction", json={"student_id": 12, "topic": 2, "action": "already_known"})
    shown = [r["topic"] for r in client.get("/recommend", params={"student_id": 12, "k": 4}).json()["recommendations"]]
    assert 2 not in shown and len(shown) == 3
    assert client.post("/reset/12").json()["removed"] == 1
    shown = [r["topic"] for r in client.get("/recommend", params={"student_id": 12, "k": 4}).json()["recommendations"]]
    assert 2 in shown


def test_invalid_action_is_rejected(client):
    r = client.post("/log_interaction", json={"student_id": 10, "topic": 1, "action": "liked"})
    assert r.status_code == 422
