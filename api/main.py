"""FastAPI backend.

Week 2 status: /recommend returns MOCK recommendations (random scores, flagged
"mode": "mock"). In Week 3 it will call src/predict.py with the trained model.
/log_interaction already writes real rows to SQLite.

Run:
    uvicorn api.main:app --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from api import db

N_TOPICS = 188


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the SQLite database once at startup (creates it if needed)."""
    app.state.conn = db.connect()
    yield
    app.state.conn.close()


app = FastAPI(title="Révise Malin API", version="0.2.0-week2", lifespan=lifespan)


class Recommendation(BaseModel):
    """One topic to revise."""

    rank: int
    topic: int
    label: str
    risk: float
    reason: str


class RecommendResponse(BaseModel):
    """Ranked topics for one student."""

    student_id: int
    mode: Literal["mock", "model"]
    recommendations: list[Recommendation]


class InteractionIn(BaseModel):
    """A feedback click sent by the interface."""

    student_id: int = Field(ge=0)
    topic: int = Field(ge=0, lt=N_TOPICS)
    action: Literal["revised", "already_known", "not_useful"]


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(student_id: int = Query(ge=0), k: int = Query(10, ge=1, le=50)) -> RecommendResponse:
    """Return the k topics with the highest risk (MOCK scores in Week 2)."""
    rng = np.random.default_rng(student_id)
    risk = rng.random(N_TOPICS)
    top = np.argsort(-risk)[:k]
    recs = [
        Recommendation(
            rank=i + 1,
            topic=int(t),
            label=f"Topic {int(t)}",
            risk=round(float(risk[t]), 3),
            reason="Mock score: the trained model is plugged in Week 3.",
        )
        for i, t in enumerate(top)
    ]
    return RecommendResponse(student_id=student_id, mode="mock", recommendations=recs)


@app.post("/log_interaction")
def log_interaction(item: InteractionIn) -> dict:
    """Store a feedback click in SQLite and return the saved row."""
    return db.log_interaction(app.state.conn, item.student_id, item.topic, item.action)


@app.get("/interactions/{student_id}")
def interactions(student_id: int) -> list[dict]:
    """List a student's stored clicks (used to show the logging works)."""
    return db.list_interactions(app.state.conn, student_id)
