"""FastAPI backend of Révise Malin.

/recommend calls the trained model (src/predict.py, loaded once at startup from
artifacts/serving.npz) to rank the DBE-KT22 knowledge components a student is
most likely to fail. Feedback sent to /log_interaction ("revised",
"already_known", "not_useful") is stored in SQLite, and the topic leaves the
student's list until the feedback is reset. /questions serves real DBE-KT22
questions of a topic and /answer checks and stores an answer; answers given in
the app move the topic in the next /recommend (src/predict.py, PRIOR_ANSWERS).

Run:
    python run.py                       (API + Streamlit interface)
    uvicorn api.main:app --port 8000    (API only)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api import db
from src.config import QUESTIONS_FILE
from src.predict import SERVING_FILE, Predictor
from src.questions import QuestionBank


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the SQLite database and load the model once at startup.

    The model file can be overridden with REVISE_MALIN_SERVING (used by the tests).
    """
    path = os.environ.get("REVISE_MALIN_SERVING", str(SERVING_FILE))
    app.state.predictor = Predictor.load(path)
    pred = app.state.predictor
    questions = Path(os.environ.get("REVISE_MALIN_QUESTIONS", str(QUESTIONS_FILE)))
    # Practice mode needs the question text, built locally from the dataset (run.py does it on first launch).
    app.state.questions = (QuestionBank.load(questions, [pred.topic_label(t) for t in range(pred.n_topics)])
                           if questions.exists() else None)
    app.state.conn = db.connect()
    yield
    app.state.conn.close()


app = FastAPI(title="Révise Malin API", version="1.0.0", lifespan=lifespan)


class Recommendation(BaseModel):
    """One topic to revise, with the numbers behind it."""

    rank: int
    topic: int
    label: str
    description: str
    related: list[str] = Field(description="Knowledge components linked to this one in DBE-KT22")
    score: float
    priority: float = Field(description="Percentile of the topic among all topics for this student (0-100)")
    graph_score: float = Field(description="Part of the score from the graph embeddings (NLGCL)")
    difficulty_score: float = Field(0.0, description="Part of the score from the learned difficulty of the topic")
    time_score: float = Field(description="Part of the score from the learned time term (beta, half-life)")
    attempts: int
    errors: int
    days_since_practice: float | None
    session_answers: int = 0
    session_errors: int = 0
    reason: str
    practice_questions: int = Field(0, description="Self-contained practice questions of the topic left for this "
                                                   "student (0 when the questions need the course worksheet)")


class RecommendResponse(BaseModel):
    """Ranked topics for one student."""

    student_id: int
    mode: Literal["model"]
    model: str
    days_ahead: float
    latency_ms: float
    session_answers: int = Field(0, description="Practice answers given in the app, which move the ranking")
    recommendations: list[Recommendation]


class InteractionIn(BaseModel):
    """A feedback click sent by the interface."""

    student_id: int = Field(ge=0)
    topic: int = Field(ge=0)
    action: Literal["revised", "already_known", "not_useful"]


class AnswerIn(BaseModel):
    """A practice question answered in the app."""

    student_id: int = Field(ge=0)
    question: int = Field(ge=0)
    choice: int = Field(ge=0)


def _predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def _check_student(pred: Predictor, student_id: int) -> None:
    if student_id >= pred.n_users:
        raise HTTPException(404, f"student_id must be between 0 and {pred.n_users - 1}")


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness check, with the model that is loaded."""
    pred = _predictor(request)
    return {"status": "ok", "students": pred.n_users, "topics": pred.n_topics, "model": pred.meta,
            "practice": request.app.state.questions is not None}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    request: Request,
    student_id: int = Query(ge=0),
    k: int = Query(10, ge=1, le=50),
    days_ahead: float = Query(0.0, ge=0, le=365, description="Simulate revising this many days from now"),
    practised_only: bool = Query(False, description="Only rank topics the student already answered"),
) -> RecommendResponse:
    """Return the k topics the student is most likely to fail, from the trained model."""
    start = time.perf_counter()
    pred = _predictor(request)
    _check_student(pred, student_id)
    hidden = db.dismissed_topics(request.app.state.conn, student_id)
    session = db.session_by_topic(request.app.state.conn, student_id)
    recs = pred.recommend(student_id, k=k, days_ahead=days_ahead, practised_only=practised_only, hidden=hidden,
                          session=session)
    bank = request.app.state.questions
    if bank is not None:
        done = {a["question"] for a in db.list_answers(request.app.state.conn, student_id)}
        for r in recs:
            r["practice_questions"] = bank.count_for_topic(r["topic"], exclude=done)
    return RecommendResponse(
        student_id=student_id, mode="model", model=pred.meta.get("model", "?"), days_ahead=days_ahead,
        latency_ms=round(1000 * (time.perf_counter() - start), 2),
        session_answers=len(db.list_answers(request.app.state.conn, student_id)),
        recommendations=[Recommendation(**r) for r in recs],
    )


@app.get("/students/{student_id}")
def student_profile(request: Request, student_id: int) -> dict:
    """History summary of one student (answers, error rate per difficulty, weakest topics)."""
    pred = _predictor(request)
    _check_student(pred, student_id)
    return pred.profile(student_id)


@app.post("/log_interaction")
def log_interaction(request: Request, item: InteractionIn) -> dict:
    """Store a feedback click in SQLite and return the saved row."""
    pred = _predictor(request)
    _check_student(pred, item.student_id)
    if item.topic >= pred.n_topics:
        raise HTTPException(422, f"topic must be between 0 and {pred.n_topics - 1}")
    return db.log_interaction(request.app.state.conn, item.student_id, item.topic, item.action)


@app.get("/interactions/{student_id}")
def interactions(request: Request, student_id: int) -> list[dict]:
    """List a student's stored clicks, newest first."""
    return db.list_interactions(request.app.state.conn, student_id)


@app.delete("/interactions/{student_id}")
def reset_interactions(request: Request, student_id: int) -> dict:
    """Delete a student's clicks (used to reset the demo)."""
    return {"deleted": db.clear_interactions(request.app.state.conn, student_id)}


def _bank(request: Request) -> QuestionBank:
    bank = request.app.state.questions
    if bank is None:
        raise HTTPException(503, "Practice questions are not built: run `python -m src.questions` (needs the dataset).")
    return bank


@app.get("/questions")
def questions(request: Request, student_id: int = Query(ge=0), topic: int = Query(ge=0),
              n: int = Query(1, ge=1, le=10),
              include_context: bool = Query(False, description="Also return questions that need the course "
                                                                "worksheet tables, which DBE-KT22 does not include")
              ) -> list[dict]:
    """Real DBE-KT22 questions of a topic to practise, without their answers; ones answered here are skipped."""
    pred = _predictor(request)
    _check_student(pred, student_id)
    if topic >= pred.n_topics:
        raise HTTPException(422, f"topic must be between 0 and {pred.n_topics - 1}")
    done = {a["question"] for a in db.list_answers(request.app.state.conn, student_id)}
    return _bank(request).for_topic(topic, n=n, exclude=done, include_context=include_context)


@app.post("/answer")
def answer(request: Request, item: AnswerIn) -> dict:
    """Check a practice answer, store it in SQLite and return the correction."""
    _check_student(_predictor(request), item.student_id)
    try:
        result = _bank(request).check(item.question, item.choice)
    except KeyError as err:
        raise HTTPException(422, str(err)) from err
    row = db.log_answer(request.app.state.conn, item.student_id, item.question, item.choice, result["correct"],
                        result["topics"])
    return {**result, "id": row["id"], "created_at": row["created_at"]}


@app.delete("/answers/{student_id}")
def reset_answers(request: Request, student_id: int) -> dict:
    """Delete a student's practice answers (used to reset the demo)."""
    return {"deleted": db.clear_answers(request.app.state.conn, student_id)}
