"""SQLite storage for what students do in the app, created automatically on first use.

Two tables:
- interactions: feedback clicks on a recommended skill (revised, already known, not useful).
- answers: questions answered in the app, with the chosen choice and whether it was correct.
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
    """Open the database and create the tables if they do not exist.

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
    """Insert one feedback click and return the stored row."""
    cur = conn.execute(
        "INSERT INTO interactions (student_id, topic, action) VALUES (?, ?, ?)",
        (student_id, topic, action),
    )
    conn.commit()
    return dict(conn.execute("SELECT * FROM interactions WHERE id = ?", (cur.lastrowid,)).fetchone())


def list_interactions(conn: sqlite3.Connection, student_id: int) -> list[dict]:
    """All clicks of one student, newest first."""
    rows = conn.execute(
        "SELECT * FROM interactions WHERE student_id = ? ORDER BY id DESC", (student_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def log_answer(conn: sqlite3.Connection, student_id: int, question: int, choice: int, correct: bool,
               topics: list[int]) -> dict:
    """Insert one answer given in the app and return the stored row."""
    cur = conn.execute(
        "INSERT INTO answers (student_id, question, choice, correct, topics) VALUES (?, ?, ?, ?, ?)",
        (student_id, question, choice, int(correct), json.dumps(topics)),
    )
    conn.commit()
    row = dict(conn.execute("SELECT * FROM answers WHERE id = ?", (cur.lastrowid,)).fetchone())
    row["topics"] = json.loads(row["topics"])
    return row


def list_answers(conn: sqlite3.Connection, student_id: int) -> list[dict]:
    """All answers of one student in the app, oldest first."""
    rows = conn.execute("SELECT * FROM answers WHERE student_id = ? ORDER BY id", (student_id,)).fetchall()
    return [{**dict(r), "topics": json.loads(r["topics"])} for r in rows]


def reset_student(conn: sqlite3.Connection, student_id: int) -> int:
    """Delete a student's app answers and clicks (to replay a demo); returns the number of rows removed."""
    n = conn.execute("DELETE FROM answers WHERE student_id = ?", (student_id,)).rowcount
    n += conn.execute("DELETE FROM interactions WHERE student_id = ?", (student_id,)).rowcount
    conn.commit()
    return n
