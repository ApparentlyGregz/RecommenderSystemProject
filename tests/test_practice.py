"""Corrected label, practice questions and answers that move the ranking."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.build_graph import window_labels
from src.predict import Predictor
from src.questions import question_blocks
from tests.test_predict import N_TOPICS, fake_serving


def test_above_mean_label_does_not_reward_exposure():
    # One student, 10 answers: topic 0 seen 8 times with 2 errors (25%), topic 1 seen twice with 1 error (50%).
    rows = [(0, t, q, 0, 0 if i < 2 else 1) for i, (t, q) in enumerate([(0, q) for q in range(8)])]
    rows += [(0, 1, 100, 0, 0), (0, 1, 101, 0, 1)]
    w = pd.DataFrame(rows, columns=["user", "topic", "question", "timestamp", "correct"])
    w["timestamp"] = np.arange(len(w))
    any_fail = window_labels(w, "any_fail").set_index("topic")["fail"]
    above = window_labels(w, "above_mean").set_index("topic")["fail"]
    assert any_fail.tolist() == [1, 1]  # both failed at least once
    assert above.tolist() == [0, 1]  # window error rate 30%: only topic 1 is above it


def test_wrong_answer_raises_priority_correct_lowers_it(tmp_path):
    pred = Predictor.load(fake_serving(tmp_path / "s.npz"))
    base = {r["topic"]: r["priority"] for r in pred.recommend(2, k=N_TOPICS)}
    t = sorted(base, key=base.get)[N_TOPICS // 2]  # a topic in the middle of the ranking
    wrong = {r["topic"]: r["priority"] for r in pred.recommend(2, k=N_TOPICS, session={t: (1, 1)})}
    right = {r["topic"]: r["priority"] for r in pred.recommend(2, k=N_TOPICS, session={t: (1, 0)})}
    assert wrong[t] > base[t] > right[t]


def test_relational_algebra_becomes_latex():
    html = 'How many rows?<br><img src="http://latex.codecogs.com/gif.latex?\\bf \\large \\color{red} {\\pi_{a}(R)}" border="0"/>'
    blocks = question_blocks(html)
    assert [b["type"] for b in blocks] == ["html", "latex"] and blocks[1]["content"] == "\\pi_{a}(R)"
    inline = question_blocks('Transactions <img src="http://latex.codecogs.com/gif.latex?T_{1}" border="0"/> run')
    assert inline == [{"type": "html", "content": "Transactions T<sub>1</sub> run"}]


@pytest.fixture
def client(tmp_path, monkeypatch):
    choices = [{"id": 10 + i, "text": f"choice {i}", "is_correct": i == 1} for i in range(3)]
    pd.DataFrame({
        "question": [1, 2], "title": ["Q1", "Q2"],
        "blocks": [json.dumps([{"type": "html", "content": "What is a join?"}])] * 2,
        "text": ["What is a join?"] * 2, "hint": ["", "think"], "explanation": ["Because.", ""],
        "difficulty": [1, 2], "choices": [json.dumps(choices)] * 2,
        "topics": [json.dumps([4]), json.dumps([4, 5])], "self_contained": [False, True],
    }).to_parquet(tmp_path / "questions.parquet")
    monkeypatch.setenv("REVISE_MALIN_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("REVISE_MALIN_SERVING", str(fake_serving(tmp_path / "serving.npz")))
    monkeypatch.setenv("REVISE_MALIN_QUESTIONS", str(tmp_path / "questions.parquet"))
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_questions_hide_answers_and_skip_worksheet_questions(client):
    assert client.get("/health").json()["practice"] is True
    qs = client.get("/questions", params={"student_id": 3, "topic": 4, "n": 2}).json()
    assert [q["question"] for q in qs] == [2]  # question 1 needs the course worksheet: not offered
    assert all("is_correct" not in c for q in qs for c in q["choices"])
    both = client.get("/questions", params={"student_id": 3, "topic": 4, "n": 2, "include_context": True}).json()
    assert [q["question"] for q in both] == [2, 1] and both[1]["needs_context"] is True


def test_recommend_says_how_many_questions_can_be_practised(client):
    def counts():
        recs = client.get("/recommend", params={"student_id": 3, "k": N_TOPICS}).json()["recommendations"]
        return {r["topic"]: r["practice_questions"] for r in recs}

    before = counts()
    assert before[4] == 1 and before[5] == 1 and before[0] == 0  # topic 4: question 1 needs the worksheet
    client.post("/answer", json={"student_id": 3, "question": 2, "choice": 11})
    after = counts()
    assert after[4] == 0 and after[5] == 0  # answered here, nothing left to practise


def test_answer_is_checked_stored_and_moves_the_topic(client):
    def priority(topic):
        recs = client.get("/recommend", params={"student_id": 3, "k": N_TOPICS}).json()["recommendations"]
        return next(r["priority"] for r in recs if r["topic"] == topic)

    before = priority(5)
    r = client.post("/answer", json={"student_id": 3, "question": 2, "choice": 10}).json()
    assert r["correct"] is False and r["correct_choices"] == [11] and r["topics"] == [4, 5]
    assert priority(5) > before
    body = client.get("/recommend", params={"student_id": 3, "k": N_TOPICS}).json()
    assert body["session_answers"] == 1
    assert next(x for x in body["recommendations"] if x["topic"] == 5)["session_errors"] == 1
    assert client.get("/questions", params={"student_id": 3, "topic": 5}).json() == []  # answered here
    assert client.delete("/answers/3").json()["deleted"] == 1


def test_bad_choice_is_rejected(client):
    assert client.post("/answer", json={"student_id": 3, "question": 1, "choice": 99}).status_code == 422
