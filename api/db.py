"""SQLite storage for what students do in the app, created automatically on first use.

- interactions: feedback clicks on a recommended notion (revised, already known, not useful).
- answers: practice questions answered in the app, with the chosen choice and whether it was correct.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    topic INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('revised', 'already_known', 'not_useful')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    question INTEGER NOT NULL,
    choice INTEGER NOT NULL,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    topics TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database and create the table if it does not exist.

    The path can be overridden with the REVISE_MALIN_DB environment variable (used by the tests).
    """
    path = Path(path or os.environ.get("REVISE_MALIN_DB", DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI runs sync endpoints in a thread pool, so the connection is shared across threads.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_interaction(conn: sqlite3.Connection, student_id: int, topic: int, action: str) -> dict:
    """Insert one click and return the stored row."""
    cur = conn.execute(
        "INSERT INTO interactions (student_id, topic, action) VALUES (?, ?, ?)",
        (student_id, topic, action),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM interactions WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_interactions(conn: sqlite3.Connection, student_id: int) -> list[dict]:
    """All clicks of one student, newest first."""
    rows = conn.execute(
        "SELECT * FROM interactions WHERE student_id = ? ORDER BY id DESC", (student_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def dismissed_topics(conn: sqlite3.Connection, student_id: int) -> set[int]:
    """Topics the student gave feedback on: they leave the recommendation list."""
    rows = conn.execute("SELECT DISTINCT topic FROM interactions WHERE student_id = ?", (student_id,)).fetchall()
    return {int(r["topic"]) for r in rows}


def clear_interactions(conn: sqlite3.Connection, student_id: int) -> int:
    """Delete every click of one student and return how many rows were removed."""
    cur = conn.execute("DELETE FROM interactions WHERE student_id = ?", (student_id,))
    conn.commit()
    return cur.rowcount


def log_answer(conn: sqlite3.Connection, student_id: int, question: int, choice: int, correct: bool,
               topics: list[int]) -> dict:
    """Insert one practice answer and return the stored row."""
    cur = conn.execute(
        "INSERT INTO answers (student_id, question, choice, correct, topics) VALUES (?, ?, ?, ?, ?)",
        (student_id, question, choice, int(correct), json.dumps(topics)),
    )
    conn.commit()
    row = dict(conn.execute("SELECT * FROM answers WHERE id = ?", (cur.lastrowid,)).fetchone())
    return {**row, "topics": json.loads(row["topics"])}


def list_answers(conn: sqlite3.Connection, student_id: int) -> list[dict]:
    """Practice answers of one student, oldest first."""
    rows = conn.execute("SELECT * FROM answers WHERE student_id = ? ORDER BY id", (student_id,)).fetchall()
    return [{**dict(r), "topics": json.loads(r["topics"])} for r in rows]


def session_by_topic(conn: sqlite3.Connection, student_id: int) -> dict[int, tuple[int, int]]:
    """Practice answers per topic: (answers, wrong answers)."""
    out: dict[int, list[int]] = {}
    for a in list_answers(conn, student_id):
        for t in a["topics"]:
            n, w = out.get(t, [0, 0])
            out[t] = [n + 1, w + (not a["correct"])]
    return {t: (n, w) for t, (n, w) in out.items()}


def clear_answers(conn: sqlite3.Connection, student_id: int) -> int:
    """Delete every practice answer of one student and return how many rows were removed."""
    cur = conn.execute("DELETE FROM answers WHERE student_id = ?", (student_id,))
    conn.commit()
    return cur.rowcount
