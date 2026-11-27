"""FastAPI backend: real recommendations from the trained model, questions, answers and feedback.

The model is loaded once at startup from artifacts/dbe/serving.npz (NumPy, no
PyTorch). Everything a student does in the app is stored in SQLite and taken
into account by the next /recommend call.

Run:
    uvicorn api.main:app --port 8000      (or python run.py for the API and the interface)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from api import db
from src.config import QUESTIONS_FILE, SERVING_FILE, TOPICS_FILE
from src.predict import Recommender, SessionAnswer


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model bundle and open the SQLite database once at startup."""
    bundle = _path("REVISE_MALIN_SERVING", SERVING_FILE)
    topics = _path("REVISE_MALIN_TOPICS", TOPICS_FILE)
    questions = _path("REVISE_MALIN_QUESTIONS", QUESTIONS_FILE)
    missing = [str(p) for p in (bundle, topics, questions) if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing files {missing}. Run `python -m src.prepare_dbe` first (see README).")
    app.state.rec = Recommender.load(bundle, topics, questions)
    app.state.conn = db.connect()
    yield
    app.state.conn.close()


app = FastAPI(title="Révise Malin API", version="1.0.0", lifespan=lifespan)


class InteractionIn(BaseModel):
    """A feedback click on a recommended skill."""

    student_id: int = Field(ge=0)
    topic: int = Field(ge=0)
    action: Literal["revised", "already_known", "not_useful"]


class AnswerIn(BaseModel):
    """A question answered in the app."""

    student_id: int = Field(ge=0)
    question: int = Field(ge=0)
    choice: int = Field(ge=0)


def _student(student_id: int) -> Recommender:
    rec: Recommender = app.state.rec
    if not rec.has_student(student_id):
        raise HTTPException(404, f"Unknown student {student_id}; see /students")
    return rec


def _topic(topic: int) -> None:
    if not 0 <= topic < len(app.state.rec.topics):
        raise HTTPException(422, f"Unknown topic {topic}")


@app.get("/health")
def health() -> dict:
    """Liveness check, with the model that is served."""
    rec = app.state.rec
    return {"status": "ok", "model": rec.meta["model"], "model_name": rec.meta["name"],
            "students": len(rec.bundle["user_ids"]), "skills": len(rec.topics)}


@app.get("/students")
def students(limit: int = Query(50, ge=1, le=1000)) -> list[dict]:
    """Students of the dataset, the most active first."""
    return sorted(app.state.rec.students(), key=lambda s: -s["answers"])[:limit]


@app.get("/recommend")
def recommend(student_id: int = Query(ge=0), k: int = Query(5, ge=1, le=30)) -> dict:
    """The k skills this student should revise first, ranked by the trained model."""
    start = time.perf_counter()
    rec = _student(student_id)
    conn = app.state.conn
    session = [SessionAnswer(a["question"], a["topics"], bool(a["correct"])) for a in db.list_answers(conn, student_id)]
    clicks = db.list_interactions(conn, student_id)
    revised = {c["topic"] for c in clicks if c["action"] == "revised"}
    hidden = {c["topic"] for c in clicks if c["action"] in {"already_known", "not_useful"}}
    recs = rec.recommend(student_id, k, session=session, revised=revised, hidden=hidden)
    return {
        "student_id": student_id, "mode": "model", "model": rec.meta["model"], "recommendations": recs,
        "session_answers": len(session), "hidden_skills": sorted(hidden),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }


@app.get("/questions")
def questions(student_id: int = Query(ge=0), topic: int = Query(ge=0), n: int = Query(3, ge=1, le=10)) -> list[dict]:
    """Real questions of a skill to practise, the ones the student has not answered yet first."""
    rec = _student(student_id)
    _topic(topic)
    done_here = {a["question"] for a in db.list_answers(app.state.conn, student_id)}
    return rec.questions_for(student_id, topic, n=n, exclude=done_here)


@app.post("/answer")
def answer(item: AnswerIn) -> dict:
    """Check an answer, store it, and return the correction."""
    rec = _student(item.student_id)
    try:
        result = rec.check(item.question, item.choice)
    except KeyError as err:
        raise HTTPException(422, str(err)) from err
    row = db.log_answer(app.state.conn, item.student_id, item.question, item.choice, result["correct"], result["topics"])
    return {**result, "id": row["id"], "created_at": row["created_at"]}


@app.post("/log_interaction")
def log_interaction(item: InteractionIn) -> dict:
    """Store a feedback click in SQLite and return the saved row."""
    _student(item.student_id)
    _topic(item.topic)
    return db.log_interaction(app.state.conn, item.student_id, item.topic, item.action)


@app.get("/interactions/{student_id}")
def interactions(student_id: int) -> dict:
    """A student's stored clicks and answers."""
    conn = app.state.conn
    return {"clicks": db.list_interactions(conn, student_id), "answers": db.list_answers(conn, student_id)}


@app.post("/reset/{student_id}")
def reset(student_id: int) -> dict:
    """Forget a student's app activity, to replay the demo from the dataset history."""
    return {"removed": db.reset_student(app.state.conn, student_id)}
