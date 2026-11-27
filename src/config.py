"""Shared constants and paths for Révise Malin.

Two datasets share the same pipeline. Pick one with the REVISE_MALIN_DATASET
environment variable (default "dbe"):

- "dbe": DBE-KT22, a database course with question text and named skills. Used by
  the app and the final results.
- "ednet": EdNet-KT1 (TOEIC), used for the Week 2 and Week 3 reports.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("REVISE_MALIN_DATASET", "dbe").lower()
if DATASET not in {"dbe", "ednet"}:
    raise ValueError(f"REVISE_MALIN_DATASET must be 'dbe' or 'ednet', not {DATASET!r}")

DATA_RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "feedback.db"
SEED = 42
MS_PER_DAY = 86_400_000
N_WINDOWS = 3  # train, validation, test label windows

if DATASET == "dbe":
    DATA_PROCESSED = ROOT / "data" / "processed" / "dbe"
    ARTIFACTS = ROOT / "artifacts" / "dbe"
    WINDOW = 30  # answers per label window (a student answers about 150 questions in total)
    OLD_TOPIC_DAYS = 7.0  # most students work in bursts of a few days
    # Label of a topic in a window: "above_mean" = the student's error rate on the topic is above
    # their own error rate over the whole window. It does not grow with how often the topic is shown.
    LABEL = "above_mean"
else:
    DATA_PROCESSED = ROOT / "data" / "processed"
    ARTIFACTS = ROOT / "artifacts"
    WINDOW = 100
    OLD_TOPIC_DAYS = 14.0
    LABEL = "any_fail"  # failed at least once in the window (Week 2 and Week 3 protocol)

EVENTS_FILE = DATA_PROCESSED / "events.parquet"
TOPICS_FILE = DATA_PROCESSED / "topics.parquet"
QUESTIONS_FILE = DATA_PROCESSED / "questions.parquet"  # DBE only: text, choices, skills
STATS_FILE = DATA_PROCESSED / "stats.json"
SERVING_FILE = ARTIFACTS / "serving.npz"  # embeddings and parameters loaded by the API

MIN_HISTORY = WINDOW  # answers that must exist before the first (training) cut
MIN_ANSWERS = WINDOW * N_WINDOWS + MIN_HISTORY  # 120 for DBE-KT22, 400 for EdNet
N_STUDENTS = 5000  # EdNet sample size

# Cut name -> number of answers from the end where the label window starts.
CUTS = {"train": 3 * WINDOW, "val": 2 * WINDOW, "test": WINDOW}
