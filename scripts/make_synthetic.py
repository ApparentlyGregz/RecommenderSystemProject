"""Generate a small fake dataset in the exact EdNet-KT1 + Contents format.

It lets anyone test the whole pipeline without downloading EdNet (1.2 GB).
The fake students forget: the chance of failing a topic rises with the time
since they last practised it.

Usage:
    python scripts/make_synthetic.py --students 300 --out data/synthetic
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

N_QUESTIONS = 1200
N_TAGS = 188
START_MS = 1_546_300_800_000  # 2019-01-01


def make_questions(rng: np.random.Generator) -> pd.DataFrame:
    """Build a questions.csv table with the EdNet Contents columns."""
    rows = []
    for q in range(1, N_QUESTIONS + 1):
        part = int(rng.integers(1, 8))
        n_tags = int(rng.integers(1, 4))
        tags = rng.choice(N_TAGS, size=n_tags, replace=False)
        rows.append(
            {
                "question_id": f"q{q}",
                "bundle_id": f"b{(q + 2) // 3}",
                "explanation_id": f"e{(q + 2) // 3}",
                "correct_answer": "abcd"[int(rng.integers(0, 4))],
                "part": part,
                "tags": ";".join(str(t) for t in tags),
                "deployed_at": START_MS,
            }
        )
    rows[0]["tags"] = "-1"  # EdNet contains a few questions without a usable tag
    return pd.DataFrame(rows)


def make_student(rng: np.random.Generator, questions: pd.DataFrame, n_answers: int) -> str:
    """Return one student's KT1 CSV as text."""
    weakness = rng.normal(0, 1, N_TAGS)
    q_tags = [
        [int(t) for t in s.split(";") if t != "-1"] for s in questions["tags"]
    ]
    exposure = rng.dirichlet(np.ones(N_QUESTIONS) * 0.3)
    t = START_MS + int(rng.integers(0, 200)) * 86_400_000
    last_seen: dict[int, int] = {}
    lines = ["timestamp,solving_id,question_id,user_answer,elapsed_time"]
    for i in range(n_answers):
        # Sessions of ~20 answers, with gaps of hours to weeks between them.
        if i % 20 == 0:
            t += int(rng.exponential(5) * 86_400_000)
        t += int(rng.integers(20, 90)) * 1000
        q = int(rng.choice(N_QUESTIONS, p=exposure))
        tags = q_tags[q] or [0]
        gaps = [(t - last_seen.get(g, t - 30 * 86_400_000)) / 86_400_000 for g in tags]
        logit = -0.8 + 0.9 * np.mean(weakness[tags]) + 1.5 * (1 - 2 ** (-np.mean(gaps) / 7))
        failed = rng.random() < 1 / (1 + np.exp(-logit))
        correct = questions.at[q, "correct_answer"]
        answer = correct if not failed else "abcd".replace(correct, "")[int(rng.integers(0, 3))]
        lines.append(f"{t},{i + 1},q{q + 1},{answer},{int(rng.integers(5, 60)) * 1000}")
        for g in tags:
            last_seen[g] = t
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write KT1.zip and questions.csv into the output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    questions = make_questions(rng)
    questions.to_csv(args.out / "questions.csv", index=False)

    with zipfile.ZipFile(args.out / "KT1.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for u in range(1, args.students + 1):
            # About a third of students are too short and must be filtered out.
            n = int(rng.integers(50, 300)) if rng.random() < 0.33 else int(rng.integers(400, 1200))
            zf.writestr(f"KT1/u{u}.csv", make_student(rng, questions, n))
    print(f"Wrote {args.students} students to {args.out / 'KT1.zip'}")


if __name__ == "__main__":
    main()
