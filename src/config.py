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

WINDOW = 100  # answers per label window
N_WINDOWS = 3  # train, validation, test label windows
MIN_HISTORY = 100  # answers that must exist before the first (training) cut
MIN_ANSWERS = WINDOW * N_WINDOWS + MIN_HISTORY  # 400
N_STUDENTS = 5000
SEED = 42

MS_PER_DAY = 86_400_000
OLD_TOPIC_DAYS = 14.0

# Cut name -> number of answers from the end where the label window starts.
CUTS = {"train": 300, "val": 200, "test": 100}
