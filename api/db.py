"""SQLite storage for feedback clicks, created automatically on first use."""
from __future__ import annotations

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
