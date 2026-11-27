"""Qualitative sanity check for the Week 3 report: the top 10 of three test students.

For each student: the model's top 10 among the topics attempted in the test window,
with the past evidence behind each topic (errors, attempts, days since practice,
answers in the last 100) and whether it was really failed. Hits are compared with
the personal-error and recent-activity baselines on the same candidates.

Usage:
    python scripts/sample_users.py --model artifacts/model_full.pt
    ->  reports/sample_users_week3.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.build_graph import build_cut  # noqa: E402
from src.config import EVENTS_FILE, TOPICS_FILE  # noqa: E402
from src.evaluate import baseline_scores, recent_activity_scores  # noqa: E402
from src.evaluate_test import LABELS, load_model  # noqa: E402
from src.model import GraphTensors  # noqa: E402

PARTS = {1: "Part 1 Photographs", 2: "Part 2 Question-response", 3: "Part 3 Conversations", 4: "Part 4 Talks",
         5: "Part 5 Incomplete sentences", 6: "Part 6 Text completion", 7: "Part 7 Reading"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "sample_users_week3.md")
    args = parser.parse_args()

    events = pd.read_parquet(EVENTS_FILE)
    topics = pd.read_parquet(TOPICS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, len(topics)
    cut = build_cut(events, "test", n_users, n_topics)
    tag, model = load_model(args.model, n_users, n_topics)
    S = model.score_all(GraphTensors.from_cut(cut)).numpy()
    recent = recent_activity_scores(events, cut)
    personal = baseline_scores(cut, "personal_error")
    popular = baseline_scores(cut, "most_popular")
    glob = popular[0]

    elig = cut.eligible()
    users = np.random.default_rng(args.seed).choice(sorted(elig), 3, replace=False)
    lines = [f"# Three test students, ranked by {LABELS.get(tag, tag)}", "",
             "Candidates = topics the student attempted in the test window (last 100 answers). "
             "\"Failed\" = got at least one question of the topic wrong in that window. "
             "Every other column only uses answers before the test cut.", ""]
    for u in users:
        cand, failed = elig[int(u)]
        order = np.argsort(-S[u, cand], kind="stable")[:10]
        top, hit = cand[order], failed[order]

        def hits(scores: np.ndarray) -> int:
            return int(failed[np.argsort(-scores[u, cand], kind="stable")[:10]].sum())

        uid = int(events.loc[events["user"] == u, "user_id"].iloc[0])
        lines += [f"## Student {int(u)} (EdNet u{uid})", "",
                  f"{len(cand)} topics attempted in the test window, {int(failed.sum())} failed. "
                  f"Failed topics in the top 10: model **{int(hit.sum())}/10**, most-popular {hits(popular)}/10, "
                  f"personal error rate {hits(personal)}/10, recent activity {hits(recent)}/10, "
                  f"random about {10 * failed.mean():.1f}/10.", "",
                  "| Rank | Topic | TOEIC part | Past errors / attempts | Days since practice | Answers in last 100 "
                  "| Most-popular rank | Failed in test |", "|---|---|---|---|---|---|---|---|"]
        for r, (t, h) in enumerate(zip(top, hit), 1):
            dt = cut.practice_dt[u, t]
            pop_rank = int((glob[cand] > glob[t]).sum()) + 1
            lines.append(f"| {r} | {int(topics.loc[t, 'tag'])} | {PARTS.get(int(topics.loc[t, 'main_part']), '?')} "
                         f"| {int(cut.fails[u, t])} / {int(cut.attempts[u, t])} "
                         f"| {'never' if np.isnan(dt) else f'{dt:.1f}'} | {int(np.floor(recent[u, t]))} "
                         f"| {pop_rank} / {len(cand)} | {'yes' if h else 'no'} |")
        lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
