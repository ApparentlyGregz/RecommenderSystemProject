"""DBE-KT22 -> one Parquet table of (answer, knowledge component) events.

DBE-KT22 (Abdelrahman et al., 2022) holds the answers of students of the
Database Systems course of the Australian National University (2019-2022) to
212 multiple-choice questions, each linked to one or more knowledge components
(KCs: "Normalisation", "Functional dependency", "SQL join"...). A KC is a
"topic" of Révise Malin: the thing a student should revise.

Steps:
    1. Read Transaction.csv: one row per answer, answer_state = correct or not.
    2. Keep students with at least MIN_ANSWERS answers (the three label windows need history).
    3. Join Question_KC_Relationships.csv: one row per (answer, KC).
    4. Index answers from the end of each student's history (answer_idx 0 = last answer).
    5. Save data/processed/events.parquet, topics.parquet (KC names, descriptions,
       prerequisites) and stats.json.

Usage:
    python -m src.prepare_data                       # reads data/raw/dbe-kt22/
    python -m src.prepare_data --raw path/to/dbe-kt22
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_graph import load_cuts, window_stats
from src.config import DATA_PROCESSED, DATA_RAW, EVENTS_FILE, MIN_ANSWERS, STATS_FILE, TOPICS_FILE

RAW_DIR = DATA_RAW / "dbe-kt22"


def load_answers(raw: Path) -> pd.DataFrame:
    """Read Transaction.csv: student, question, time (ms since epoch, UTC) and correctness."""
    t = pd.read_csv(raw / "Transaction.csv",
                    usecols=["student_id", "question_id", "start_time", "answer_state", "hint_used"])
    # Times look like "2019-08-07 17:12:08.722 -0700": parse with their UTC offset.
    ts = pd.to_datetime(t["start_time"], format="%Y-%m-%d %H:%M:%S.%f %z", utc=True)
    return pd.DataFrame({
        "user_id": t["student_id"].astype(np.int64),
        "question": t["question_id"].astype(np.int32),
        # Milliseconds since 1970, whatever the internal resolution pandas picked (ns or us).
        "timestamp": ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).astype(np.int64),
        "correct": t["answer_state"].astype(bool).astype(np.int8),
        "hint_used": t["hint_used"].astype(bool).astype(np.int8),
    })


def load_topics(raw: Path, used_kcs: set[int]) -> pd.DataFrame:
    """KCs that appear on at least one answered question, with names, descriptions and prerequisites."""
    kcs = pd.read_csv(raw / "KCs.csv").rename(columns={"id": "kc"})
    kcs = kcs[kcs["kc"].isin(used_kcs)].sort_values("kc").reset_index(drop=True)
    kcs["topic"] = np.arange(len(kcs), dtype=np.int16)
    # KC_Relationships: from -> to. Kept for the interface ("related notions"), not used by the model.
    rel = pd.read_csv(raw / "KC_Relationships.csv")
    rel = rel[rel["from_knowledgecomponent_id"].isin(used_kcs) & rel["to_knowledgecomponent_id"].isin(used_kcs)]
    related = rel.groupby("from_knowledgecomponent_id")["to_knowledgecomponent_id"].apply(
        lambda s: ";".join(str(int(x)) for x in sorted(set(s))))
    kcs["related_kcs"] = kcs["kc"].map(related).fillna("")
    kcs["name"] = kcs["name"].str.strip()
    kcs["description"] = kcs["description"].fillna("").str.strip()
    return kcs[["topic", "kc", "name", "description", "related_kcs"]]


def build_events(answers: pd.DataFrame, raw: Path, min_answers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter students, explode answers into (answer, KC) rows and index them.

    Args:
        answers: Output of load_answers.
        raw: Folder of the raw DBE-KT22 CSV files.
        min_answers: Minimum number of answers per student.

    Returns:
        events (one row per answer and KC) and topics (one row per KC).
    """
    counts = answers.groupby("user_id").size()
    answers = answers[answers["user_id"].isin(counts[counts >= min_answers].index)]
    answers = answers.sort_values(["user_id", "timestamp", "question"], kind="stable").copy()
    answers["answer_idx"] = answers.groupby("user_id").cumcount(ascending=False).astype(np.int32)

    q_kc = pd.read_csv(raw / "Question_KC_Relationships.csv")
    q_kc = q_kc.rename(columns={"question_id": "question", "knowledgecomponent_id": "kc"})[["question", "kc"]]
    questions = pd.read_csv(raw / "Questions.csv", usecols=["id", "difficulty"]).rename(columns={"id": "question"})
    ev = answers.merge(q_kc, on="question", how="inner").merge(questions, on="question", how="left")

    topics = load_topics(raw, set(ev["kc"].unique()))
    topics["n_events"] = topics["kc"].map(ev["kc"].value_counts()).astype(np.int64)
    ev["topic"] = ev["kc"].map(dict(zip(topics["kc"], topics["topic"]))).astype(np.int16)
    users = np.sort(ev["user_id"].unique())
    ev["user"] = ev["user_id"].map({u: i for i, u in enumerate(users)}).astype(np.int32)
    ev["difficulty"] = ev["difficulty"].fillna(0).astype(np.int8)
    cols = ["user", "user_id", "timestamp", "answer_idx", "question", "difficulty", "topic", "correct", "hint_used"]
    ev = ev[cols].sort_values(["user", "timestamp", "question", "topic"], kind="stable").reset_index(drop=True)
    return ev, topics


def main() -> None:
    """Run the preprocessing and print the counts used in the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="folder with the DBE-KT22 CSV files")
    parser.add_argument("--min-answers", type=int, default=MIN_ANSWERS)
    args = parser.parse_args()

    start = time.time()
    answers = load_answers(args.raw)
    events, topics = build_events(answers, args.raw, args.min_answers)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS_FILE, index=False, compression="zstd")
    topics.to_parquet(TOPICS_FILE, index=False)

    n_users, n_topics = int(events["user"].max()) + 1, len(topics)
    cuts = load_cuts(events, n_users, n_topics)
    one_per_answer = events.drop_duplicates(["user", "timestamp", "question"])
    stats = {
        "dataset": "DBE-KT22",
        "students_in_raw_data": int(answers["user_id"].nunique()),
        "answers_in_raw_data": int(len(answers)),
        "students": n_users,
        "answers": int(len(one_per_answer)),
        "events_answer_x_topic": int(len(events)),
        "topics": n_topics,
        "questions_seen": int(events["question"].nunique()),
        "answers_per_student_median": float(one_per_answer.groupby("user").size().median()),
        "overall_error_rate": float(1 - one_per_answer["correct"].mean()),
        "first_answer": str(pd.to_datetime(events["timestamp"].min(), unit="ms").date()),
        "last_answer": str(pd.to_datetime(events["timestamp"].max(), unit="ms").date()),
        "seconds": round(time.time() - start, 1),
        "cuts": {name: window_stats(c) for name, c in cuts.items()},
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
