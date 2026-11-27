"""FastAPI backend of Révise Malin.

/recommend calls the trained model (src/predict.py, loaded once at startup from
artifacts/serving.npz) to rank the DBE-KT22 knowledge components a student is
most likely to fail. Feedback sent to /log_interaction ("revised",
"already_known", "not_useful") is stored in SQLite, and the topic leaves the
student's list until the feedback is reset.

Run:
    python run.py                       (API + Streamlit interface)
    uvicorn api.main:app --port 8000    (API only)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api import db
from src.predict import SERVING_FILE, Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the SQLite database and load the model once at startup.

    The model file can be overridden with REVISE_MALIN_SERVING (used by the tests).
    """
    path = os.environ.get("REVISE_MALIN_SERVING", str(SERVING_FILE))
    app.state.predictor = Predictor.load(path)
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
    time_score: float = Field(description="Part of the score from the learned time term (beta, half-life)")
    attempts: int
    errors: int
    days_since_practice: float | None
    reason: str


class RecommendResponse(BaseModel):
    """Ranked topics for one student."""

    student_id: int
    mode: Literal["model"]
    model: str
    days_ahead: float
    latency_ms: float
    recommendations: list[Recommendation]


class InteractionIn(BaseModel):
    """A feedback click sent by the interface."""

    student_id: int = Field(ge=0)
    topic: int = Field(ge=0)
    action: Literal["revised", "already_known", "not_useful"]


def _predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def _check_student(pred: Predictor, student_id: int) -> None:
    if student_id >= pred.n_users:
        raise HTTPException(404, f"student_id must be between 0 and {pred.n_users - 1}")


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness check, with the model that is loaded."""
    pred = _predictor(request)
    return {"status": "ok", "students": pred.n_users, "topics": pred.n_topics, "model": pred.meta}


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
    recs = pred.recommend(student_id, k=k, days_ahead=days_ahead, practised_only=practised_only, hidden=hidden)
    return RecommendResponse(
        student_id=student_id, mode="model", model=pred.meta.get("model", "?"), days_ahead=days_ahead,
        latency_ms=round(1000 * (time.perf_counter() - start), 2),
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
