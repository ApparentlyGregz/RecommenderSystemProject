"""Small fake event table shared by the tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_TOPICS = 12


@pytest.fixture
def events() -> pd.DataFrame:
    """Three students with 450 answers each; pairs of answers share a timestamp, like a bundle."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(3):
        t = 1_546_300_800_000
        n = 450
        for i in range(n):
            if i % 2 == 0:
                t += int(rng.integers(1, 10)) * 86_400_000
            rows.append(
                {"user": u, "timestamp": t, "answer_idx": n - 1 - i,
                 "topic": int(rng.integers(0, N_TOPICS)), "correct": int(rng.random() < 0.6)}
            )
    return pd.DataFrame(rows)
