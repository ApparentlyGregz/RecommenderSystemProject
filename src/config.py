"""Shared constants and paths for Révise Malin.

Two datasets share the pipeline; REVISE_MALIN_DATASET picks one (default "dbe"):
- "dbe": DBE-KT22, the showcase dataset (named knowledge components, question text).
- "ednet": EdNet-KT1 (TOEIC), the Week 2 and Week 3 evidence (src/prepare_ednet.py).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.environ.get("REVISE_MALIN_DATASET", "dbe").lower()
if DATASET not in {"dbe", "ednet"}:
    raise ValueError(f"REVISE_MALIN_DATASET must be 'dbe' or 'ednet', not {DATASET!r}")

DATA_RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "feedback.db"
N_WINDOWS = 3  # train, validation, test label windows
SEED = 42
MS_PER_DAY = 86_400_000
OLD_TOPIC_DAYS = 14.0
N_STUDENTS = 5000  # EdNet sample size

if DATASET == "dbe":
    DATA_PROCESSED = ROOT / "data" / "processed"
    ARTIFACTS = ROOT / "artifacts"
    # DBE-KT22 students answer 148 questions (median), so the windows are shorter than the
    # 100 answers used on EdNet in Weeks 2-3. 25 answers give a median of 17 candidate
    # topics per window and keep 718 of the 1,264 students.
    WINDOW = 25  # answers per label window
    MIN_HISTORY = 50  # answers that must exist before the first (training) cut
else:
    DATA_PROCESSED = ROOT / "data" / "processed" / "ednet"
    ARTIFACTS = ROOT / "artifacts" / "ednet"
    WINDOW = 100
    MIN_HISTORY = 100

# Label of a topic attempted in a label window (src/build_graph.py, window_labels):
# - "above_mean": the student's error rate on the topic is above their own error rate over the
#   window. It asks "where is this student weaker than usual?" and does not grow with how often
#   the topic is shown. Official DBE-KT22 results.
# - "any_fail": failed at least once in the window. It grows with exposure (Week 3 report), so it
#   rewards the most-practised topics. Week 2-3 protocol, kept as a secondary result.
LABEL = os.environ.get("REVISE_MALIN_LABEL", "above_mean" if DATASET == "dbe" else "any_fail")

EVENTS_FILE = DATA_PROCESSED / "events.parquet"
TOPICS_FILE = DATA_PROCESSED / "topics.parquet"
QUESTIONS_FILE = DATA_PROCESSED / "questions.parquet"  # DBE-KT22 question text, built locally (src/questions.py)
STATS_FILE = DATA_PROCESSED / "stats.json"

MIN_ANSWERS = WINDOW * N_WINDOWS + MIN_HISTORY  # 125 on DBE-KT22, 400 on EdNet

# Cut name -> number of answers from the end where the label window starts.
CUTS = {"train": 3 * WINDOW, "val": 2 * WINDOW, "test": WINDOW}
