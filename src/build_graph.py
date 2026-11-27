"""Build model inputs and labels at a cut, with no leakage.

For each student, a cut is a point in time. Everything strictly before it is
input (graph edges, time gaps, error rates). The window right after it gives
the labels of the topics attempted in it, with one of two rules (config.LABEL):

- "any_fail": positive if failed at least once in the window (EdNet reports).
  This grows with how often a topic is shown, so it rewards exposure.
- "above_mean": positive if the student's error rate on the topic is above their
  error rate over the whole window. It asks "where is this student weaker than
  usual?", whatever the number of attempts (DBE-KT22, final results).

Cuts are defined on timestamps, not on row positions, so a bundle of questions
answered at the same moment is never split between input and label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CUTS, LABEL, MS_PER_DAY, OLD_TOPIC_DAYS, WINDOW


@dataclass
class Cut:
    """Inputs and labels for one cut (train, val or test).

    Attributes:
        name: Cut name.
        n_users: Number of students (rows of the matrices).
        n_topics: Number of topics (columns of the matrices).
        edge_user: Student index of each graph edge (failed before the cut).
        edge_topic: Topic index of each graph edge.
        edge_dt: Days between the last error on the edge and the cut.
        practice_dt: Days since last practice, shape (n_users, n_topics), NaN if never seen.
        attempts: Answers per (student, topic) before the cut.
        fails: Wrong answers per (student, topic) before the cut.
        cand_user: Student index of each label candidate (attempted in the window).
        cand_topic: Topic index of each label candidate.
        cand_failed: 1 if the candidate was failed at least once in the window.
    """

    name: str
    n_users: int
    n_topics: int
    edge_user: np.ndarray
    edge_topic: np.ndarray
    edge_dt: np.ndarray
    practice_dt: np.ndarray
    attempts: np.ndarray
    fails: np.ndarray
    cand_user: np.ndarray
    cand_topic: np.ndarray
    cand_failed: np.ndarray

    def candidates(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Group label candidates by student: {user: (topics, failed)}."""
        order = np.argsort(self.cand_user, kind="stable")
        users, starts = np.unique(self.cand_user[order], return_index=True)
        out = {}
        for u, a, b in zip(users, starts, list(starts[1:]) + [len(order)]):
            idx = order[a:b]
            out[int(u)] = (self.cand_topic[idx], self.cand_failed[idx])
        return out

    def eligible(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Students with at least one failed and one correct candidate."""
        return {u: c for u, c in self.candidates().items() if 0 < c[1].sum() < len(c[1])}


def cut_times(events: pd.DataFrame) -> pd.DataFrame:
    """Return, per student, the timestamp where each label window starts.

    Args:
        events: Event table with user, timestamp and answer_idx (0 = last answer).

    Returns:
        DataFrame indexed by user with one column per cut name.
    """
    out = {}
    for name, c in CUTS.items():
        window = events[(events["answer_idx"] < c) & (events["answer_idx"] >= c - WINDOW)]
        out[name] = window.groupby("user")["timestamp"].min()
    return pd.DataFrame(out)


def window_labels(window: pd.DataFrame, rule: str = LABEL) -> pd.DataFrame:
    """Label of each (user, topic) attempted in a label window.

    Args:
        window: Events of the label window (user, topic, correct, and question for "above_mean").
        rule: "any_fail" or "above_mean" (see the module docstring).

    Returns:
        DataFrame with user, topic and fail (0/1).
    """
    w = window.assign(fail=1 - window["correct"])
    if rule == "any_fail":
        return w.groupby(["user", "topic"])["fail"].max().reset_index()
    if rule == "above_mean":
        # The student's error rate over the window counts each answer once, even if it has several topics.
        key = ["user", "timestamp", "question"] if "question" in w else ["user", "timestamp"]
        mean = w.drop_duplicates(key).groupby("user")["fail"].mean()
        rate = w.groupby(["user", "topic"])["fail"].mean().reset_index()
        rate["fail"] = (rate["fail"] > rate["user"].map(mean) + 1e-9).astype(np.int8)
        return rate
    raise ValueError(f"Unknown label rule: {rule}")


def build_cut(events: pd.DataFrame, name: str, n_users: int, n_topics: int, rule: str = LABEL) -> Cut:
    """Build inputs and labels for one cut.

    Args:
        events: Event table (user, topic, timestamp, correct, answer_idx), indices already mapped.
        name: "train", "val" or "test".
        n_users: Number of students.
        n_topics: Number of topics.

    Returns:
        The Cut with inputs strictly before the cut and labels in the next window.
    """
    times = cut_times(events)
    names = list(CUTS)
    t_start = events["user"].map(times[name])
    nxt = names.index(name) + 1
    t_end = events["user"].map(times[names[nxt]]) if nxt < len(names) else np.inf

    past = events[events["timestamp"] < t_start]
    window = events[(events["timestamp"] >= t_start) & (events["timestamp"] < t_end)]
    return _cut_from(past, times[name], window, name, n_users, n_topics, rule)


def build_history(events: pd.DataFrame, n_users: int, n_topics: int) -> Cut:
    """Inputs built from every answer, as of each student's last answer (used to serve the app).

    There is no label window: the candidates are empty. Time gaps are measured
    from the student's last answer, so a demo replays the moment they stopped.
    """
    now = events.groupby("user")["timestamp"].max() + 1
    return _cut_from(events, now, events.iloc[:0], "now", n_users, n_topics, "any_fail")


def _cut_from(past: pd.DataFrame, t_cut: pd.Series, window: pd.DataFrame, name: str, n_users: int,
              n_topics: int, rule: str) -> Cut:
    """Aggregate the history before each student's cut time and label the window after it."""
    g = past.assign(fail=1 - past["correct"]).groupby(["user", "topic"])
    agg = g.agg(attempts=("fail", "size"), fails=("fail", "sum"), last_ts=("timestamp", "max"))
    agg = agg.reset_index()
    u, t = agg["user"].to_numpy(), agg["topic"].to_numpy()
    cut_ts = agg["user"].map(t_cut).to_numpy()

    attempts = np.zeros((n_users, n_topics), np.float32)
    fails = np.zeros((n_users, n_topics), np.float32)
    practice_dt = np.full((n_users, n_topics), np.nan, np.float32)
    attempts[u, t] = agg["attempts"].to_numpy()
    fails[u, t] = agg["fails"].to_numpy()
    practice_dt[u, t] = (cut_ts - agg["last_ts"].to_numpy()) / MS_PER_DAY

    wrong = past[past["correct"] == 0].groupby(["user", "topic"])["timestamp"].max().reset_index()
    edge_dt = (wrong["user"].map(t_cut).to_numpy() - wrong["timestamp"].to_numpy()) / MS_PER_DAY

    lab = window_labels(window, rule) if len(window) else pd.DataFrame({"user": [], "topic": [], "fail": []})
    return Cut(
        name=name,
        n_users=n_users,
        n_topics=n_topics,
        edge_user=wrong["user"].to_numpy(np.int64),
        edge_topic=wrong["topic"].to_numpy(np.int64),
        edge_dt=edge_dt.astype(np.float32),
        practice_dt=practice_dt,
        attempts=attempts,
        fails=fails,
        cand_user=lab["user"].to_numpy(np.int64),
        cand_topic=lab["topic"].to_numpy(np.int64),
        cand_failed=lab["fail"].to_numpy(np.int8),
    )


def load_cuts(events: pd.DataFrame, n_users: int, n_topics: int) -> dict[str, Cut]:
    """Build the three cuts (train, val, test)."""
    return {name: build_cut(events, name, n_users, n_topics) for name in CUTS}


def window_stats(cut: Cut) -> dict[str, float]:
    """EDA numbers that decide whether the metrics are informative.

    Args:
        cut: A built cut.

    Returns:
        Counts of eligible students, candidate list sizes and old-topic coverage.
    """
    cands = cut.candidates()
    elig = cut.eligible()
    sizes = np.array([len(c[0]) for c in elig.values()]) if elig else np.array([0])
    old_users = 0
    old_failed = 0
    failed_total = 0
    for u, (topics, failed) in elig.items():
        dt = cut.practice_dt[u, topics]
        is_old = (dt > OLD_TOPIC_DAYS) & (failed == 1)
        old_failed += int(is_old.sum())
        failed_total += int(failed.sum())
        old_users += int(is_old.any())
    return {
        "students_with_labels": len(cands),
        "eligible_students": len(elig),
        "median_candidates": float(np.median(sizes)),
        "share_more_than_20_candidates": float((sizes > 20).mean()),
        "share_failed_candidates_old": old_failed / max(failed_total, 1),
        "students_with_old_failed_topic": old_users,
        "graph_edges": int(len(cut.edge_user)),
    }
