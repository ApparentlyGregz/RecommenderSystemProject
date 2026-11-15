"""EdNet-KT1 + Contents -> one Parquet table of (answer, topic) events.

Steps:
    1. Pick students at random among those with at least MIN_ANSWERS answers.
       File sizes from the zip index pre-filter short students without unzipping.
    2. Join answers with questions.csv to compute correct = user_answer == correct_answer.
    3. Explode tags: one row per (answer, topic). Tag -1 (no tag) is dropped.
    4. Save data/processed/events.parquet, topics.parquet and stats.json.

Usage:
    python -m src.prepare_data --kt1 data/raw/EdNet-KT1.zip --questions data/raw/contents/questions.csv
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_graph import load_cuts, window_stats
from src.config import (
    DATA_PROCESSED,
    EVENTS_FILE,
    MIN_ANSWERS,
    N_STUDENTS,
    SEED,
    STATS_FILE,
    TOPICS_FILE,
)

MIN_BYTES_PER_LINE = 23  # shortest possible KT1 line, so the size filter never drops a valid student
USER_RE = re.compile(r"u(\d+)\.csv$")


def list_student_files(kt1: Path, zf: zipfile.ZipFile | None) -> list[tuple[str, int]]:
    """List (name, size in bytes) of every student CSV in the open zip, or in a folder."""
    if zf is not None:
        return [(i.filename, i.file_size) for i in zf.infolist() if USER_RE.search(i.filename)]
    return [(str(p), p.stat().st_size) for p in kt1.rglob("u*.csv")]


def read_student(kt1: Path, name: str, zf: zipfile.ZipFile | None) -> pd.DataFrame:
    """Read one student CSV from the zip or the folder."""
    if zf is not None:
        with zf.open(name) as f:
            return pd.read_csv(io.BytesIO(f.read()))
    return pd.read_csv(name)


def load_questions(path: Path) -> pd.DataFrame:
    """Load questions.csv and keep id, correct answer, part and tags."""
    q = pd.read_csv(path, usecols=["question_id", "correct_answer", "part", "tags"])
    return q.dropna(subset=["correct_answer"])


def sample_students(kt1: Path, n_students: int, min_answers: int, seed: int) -> pd.DataFrame:
    """Read random students until n_students with at least min_answers valid answers are found.

    Args:
        kt1: Path to EdNet-KT1.zip or to the unzipped folder.
        n_students: Target number of students.
        min_answers: Minimum number of answers per student.
        seed: Random seed for the sampling order.

    Returns:
        Raw answers of the selected students with a ``user`` column.
    """
    zf = zipfile.ZipFile(kt1) if kt1.suffix == ".zip" else None
    kept: list[pd.DataFrame] = []
    try:
        files = list_student_files(kt1, zf)
        min_bytes = min_answers * MIN_BYTES_PER_LINE
        candidates = [name for name, size in files if size >= min_bytes]
        del files
        rng = np.random.default_rng(seed)
        rng.shuffle(candidates)
        print(f"{len(candidates):,} student files pass the size pre-filter")

        for name in candidates:
            df = read_student(kt1, name, zf).dropna(subset=["user_answer"])
            if len(df) >= min_answers:
                df["user"] = int(USER_RE.search(name).group(1))
                kept.append(df)
                if len(kept) % 500 == 0:
                    print(f"  {len(kept):,} students kept")
                if len(kept) == n_students:
                    break
    finally:
        if zf is not None:
            zf.close()
    if len(kept) < n_students:
        print(f"Warning: only {len(kept)} students have at least {min_answers} answers")
    return pd.concat(kept, ignore_index=True)


def build_events(answers: pd.DataFrame, questions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join, compute correctness, index answers from the end and explode tags.

    Args:
        answers: Raw KT1 rows with a ``user`` column.
        questions: Output of load_questions.

    Returns:
        events (one row per answer and topic) and topics (one row per topic).
    """
    df = answers.merge(questions, on="question_id", how="inner")
    df = df.sort_values(["user", "timestamp", "solving_id"], kind="stable")
    df["correct"] = (df["user_answer"] == df["correct_answer"]).astype(np.int8)
    df["answer_idx"] = df.groupby("user").cumcount(ascending=False).astype(np.int32)

    df["tag"] = df["tags"].astype(str).str.split(";")
    ev = df.explode("tag")
    ev = ev[ev["tag"].str.fullmatch(r"\d+", na=False)]
    ev["tag"] = ev["tag"].astype(np.int16)

    topics = (
        ev.groupby("tag")
        .agg(main_part=("part", lambda s: int(s.mode().iloc[0])), n_events=("part", "size"))
        .reset_index()
        .sort_values("tag")
        .reset_index(drop=True)
    )
    topics["topic"] = np.arange(len(topics), dtype=np.int16)
    users = np.sort(ev["user"].unique())

    ev["topic"] = ev["tag"].map(dict(zip(topics["tag"], topics["topic"]))).astype(np.int16)
    ev["user_id"] = ev["user"].astype(np.int64)
    ev["user"] = ev["user"].map({u: i for i, u in enumerate(users)}).astype(np.int32)
    ev["question"] = ev["question_id"].str[1:].astype(np.int32)
    ev["part"] = ev["part"].astype(np.int8)
    cols = ["user", "user_id", "timestamp", "answer_idx", "question", "part", "topic", "correct"]
    return ev[cols].reset_index(drop=True), topics


def main() -> None:
    """Run the full preprocessing and print the counts used in the Week 2 report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kt1", type=Path, required=True, help="EdNet-KT1.zip or the unzipped KT1 folder")
    parser.add_argument("--questions", type=Path, required=True, help="Contents questions.csv")
    parser.add_argument("--students", type=int, default=N_STUDENTS)
    parser.add_argument("--min-answers", type=int, default=MIN_ANSWERS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    start = time.time()
    answers = sample_students(args.kt1, args.students, args.min_answers, args.seed)
    events, topics = build_events(answers, load_questions(args.questions))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS_FILE, index=False, compression="zstd")
    topics.to_parquet(TOPICS_FILE, index=False)

    n_users, n_topics = int(events["user"].max()) + 1, len(topics)
    cuts = load_cuts(events, n_users, n_topics)
    per_student = answers.groupby("user").size()
    stats = {
        "students": n_users,
        "answers": int(len(answers)),
        "events_answer_x_topic": int(len(events)),
        "topics": n_topics,
        "questions_seen": int(events["question"].nunique()),
        "answers_per_student_median": float(per_student.median()),
        "overall_error_rate": float(1 - events.drop_duplicates(["user", "timestamp", "question"])["correct"].mean()),
        "parquet_mb": round(EVENTS_FILE.stat().st_size / 1e6, 2),
        "seconds": round(time.time() - start, 1),
        "cuts": {name: window_stats(c) for name, c in cuts.items()},
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
