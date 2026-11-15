"""Evaluation harness: one protocol for every model and baseline.

For each eligible student (at least one failed and one correct topic in the
label window), a model ranks ONLY the topics the student attempted in that
window. Relevant = failed at least once. Ranking all 188 topics would measure
which questions the Santa app chose to show, not the risk of failing.

Metrics: NDCG@10, Recall@20 (required), Recall@5 (fallback when lists are
short) and Recall@10 on "old" failed topics (not practised for > 14 days),
the metric of our forgetting component.

Usage:
    python -m src.evaluate            # baselines + mock-prediction check on the test cut
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.build_graph import Cut, load_cuts
from src.config import EVENTS_FILE, OLD_TOPIC_DAYS, SEED, TOPICS_FILE


def ndcg_at_k(rel_in_rank_order: np.ndarray, k: int) -> float:
    """Binary NDCG@k for one ranked list."""
    gains = rel_in_rank_order[:k] / np.log2(np.arange(2, min(k, len(rel_in_rank_order)) + 2))
    n_rel = int(rel_in_rank_order.sum())
    ideal = (1 / np.log2(np.arange(2, min(k, n_rel) + 2))).sum()
    return float(gains.sum() / ideal) if ideal > 0 else 0.0


def recall_at_k(rel_in_rank_order: np.ndarray, k: int) -> float:
    """Share of the relevant items found in the top k."""
    n_rel = rel_in_rank_order.sum()
    return float(rel_in_rank_order[:k].sum() / n_rel) if n_rel > 0 else 0.0


def rank(scores: np.ndarray, tie_key: np.ndarray) -> np.ndarray:
    """Indices sorted by decreasing score; ties broken by a fixed random key."""
    return np.lexsort((tie_key, -scores))


def evaluate_scores(scores: np.ndarray, cut: Cut, seed: int = SEED) -> dict:
    """Score a (n_users, n_topics) matrix on a cut.

    Args:
        scores: Risk score of every (student, topic); higher = ranked first.
        cut: The cut whose label window is evaluated.
        seed: Seed of the tie-breaking key.

    Returns:
        Mean metrics, number of students, and per-student arrays (for confidence intervals).
    """
    tie = np.random.default_rng(seed).random(cut.n_topics)
    per = {"ndcg@10": [], "recall@20": [], "recall@5": []}
    old_recall = []
    for u, (topics, failed) in sorted(cut.eligible().items()):
        order = rank(scores[u, topics], tie[topics])
        rel = failed[order].astype(float)
        per["ndcg@10"].append(ndcg_at_k(rel, 10))
        per["recall@20"].append(recall_at_k(rel, 20))
        per["recall@5"].append(recall_at_k(rel, 5))
        old = (cut.practice_dt[u, topics] > OLD_TOPIC_DAYS) & (failed == 1)
        if old.any():
            old_recall.append(recall_at_k(old[order].astype(float), 10))
    out = {name: float(np.mean(v)) for name, v in per.items()}
    out["recall@10_old"] = float(np.mean(old_recall)) if old_recall else float("nan")
    out["students"] = len(per["ndcg@10"])
    out["students_old"] = len(old_recall)
    out["per_student"] = {name: np.array(v) for name, v in per.items()}
    return out


def baseline_scores(cut: Cut, name: str, seed: int = SEED) -> np.ndarray:
    """Baselines computed only from inputs before the cut.

    Args:
        cut: The cut.
        name: "random", "global_error" (most-popular) or "personal_error".
        seed: Seed for the random baseline.

    Returns:
        A (n_users, n_topics) score matrix.
    """
    shape = (cut.n_users, cut.n_topics)
    if name == "random":
        return np.random.default_rng(seed).random(shape)
    global_rate = (cut.fails.sum(0) + 1) / (cut.attempts.sum(0) + 2)
    if name == "global_error":
        return np.broadcast_to(global_rate, shape).copy()
    if name == "personal_error":
        m = 5.0  # shrink students' own rates towards the global rate when they have few answers
        return (cut.fails + m * global_rate) / (cut.attempts + m)
    raise ValueError(f"Unknown baseline: {name}")


def bootstrap_gap(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of a - b over students with a 95% bootstrap interval."""
    diff = a - b
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_baselines(cut: Cut, n_random_seeds: int = 10) -> pd.DataFrame:
    """Table of baseline metrics on a cut; random is averaged over several seeds."""
    rows = []
    rand = [evaluate_scores(baseline_scores(cut, "random", s), cut) for s in range(n_random_seeds)]
    rows.append({"model": f"random (mean of {n_random_seeds} seeds)", **_mean_metrics(rand)})
    for name in ["global_error", "personal_error"]:
        res = evaluate_scores(baseline_scores(cut, name), cut)
        rows.append({"model": name, **{k: v for k, v in res.items() if k != "per_student"}})
    return pd.DataFrame(rows)


def _mean_metrics(results: list[dict]) -> dict:
    keys = ["ndcg@10", "recall@20", "recall@5", "recall@10_old", "students", "students_old"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def main() -> None:
    """Check the harness on mock predictions, then print the baseline table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cut", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    events = pd.read_parquet(EVENTS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, len(pd.read_parquet(TOPICS_FILE))
    cut = load_cuts(events, n_users, n_topics)[args.cut]

    mock = np.random.default_rng(123).normal(size=(n_users, n_topics))
    res = evaluate_scores(mock, cut)
    print("Mock predictions ->", json.dumps({k: v for k, v in res.items() if k != "per_student"}))

    table = run_baselines(cut)
    print(f"\nBaselines on the {args.cut} cut:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
