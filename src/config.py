"""Shared constants and paths for Révise Malin."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
DB_PATH = ROOT / "data" / "feedback.db"

EVENTS_FILE = DATA_PROCESSED / "events.parquet"
TOPICS_FILE = DATA_PROCESSED / "topics.parquet"
STATS_FILE = DATA_PROCESSED / "stats.json"

# DBE-KT22 students answer 148 questions (median), so the windows are shorter than the
# 100 answers used on EdNet in Weeks 2-3. 25 answers give a median of 17 candidate
# topics per window and keep 718 of the 1,264 students.
WINDOW = 25  # answers per label window
N_WINDOWS = 3  # train, validation, test label windows
MIN_HISTORY = 50  # answers that must exist before the first (training) cut
MIN_ANSWERS = WINDOW * N_WINDOWS + MIN_HISTORY  # 125
SEED = 42

MS_PER_DAY = 86_400_000
OLD_TOPIC_DAYS = 14.0

# Cut name -> number of answers from the end where the label window starts.
CUTS = {"train": 3 * WINDOW, "val": 2 * WINDOW, "test": WINDOW}
