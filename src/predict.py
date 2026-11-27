"""Serving-time predictor: ranks the knowledge components a student should revise first.

Loads artifacts/serving.npz (written by src.export_serving) and recomputes, at
every request, the score of the trained model:

    score(student, topic) = <e_student, e_topic> + time_term(dt)
    time_term(dt) = beta * (1 - 2^(-dt / h_topic))   if the topic was practised dt days ago
                  = unseen                           if it was never practised

The embeddings are fixed (propagated once at export time). The time term is
cheap, so the API can move time forward ("what if I revise in 10 days?").
beta is learned: beta > 0 means a topic gets riskier as it is forgotten, beta < 0
means the topics practised right now are the ones failed next. On DBE-KT22 the
model learned beta < 0, and the explanations say so instead of talking about forgetting.
Only numpy is needed, so the API does not import PyTorch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS

SERVING_FILE = ARTIFACTS / "serving.npz"
LEVELS = {1: "Easy", 2: "Medium", 3: "Hard"}


@dataclass
class Predictor:
    """Numpy version of the trained model, plus the evidence used to explain its ranking."""

    user_emb: np.ndarray
    topic_emb: np.ndarray
    half_life: np.ndarray
    beta: float
    unseen: float
    use_forgetting: bool
    practice_dt: np.ndarray
    attempts: np.ndarray
    fails: np.ndarray
    user_id: np.ndarray
    n_answers: np.ndarray
    level_attempts: np.ndarray
    level_fails: np.ndarray
    topic_kc: np.ndarray
    topic_name: np.ndarray
    topic_description: np.ndarray
    topic_related: np.ndarray
    meta: dict

    @classmethod
    def load(cls, path: Path | str = SERVING_FILE) -> "Predictor":
        """Read the exported file."""
        with np.load(path) as z:
            return cls(
                user_emb=z["user_emb"], topic_emb=z["topic_emb"], half_life=z["half_life"],
                beta=float(z["beta"]), unseen=float(z["unseen"]), use_forgetting=bool(z["use_forgetting"]),
                practice_dt=z["practice_dt"], attempts=z["attempts"].astype(np.int64),
                fails=z["fails"].astype(np.int64), user_id=z["user_id"], n_answers=z["n_answers"],
                level_attempts=z["level_attempts"].astype(np.int64), level_fails=z["level_fails"].astype(np.int64),
                topic_kc=z["topic_kc"], topic_name=z["topic_name"], topic_description=z["topic_description"],
                topic_related=z["topic_related"], meta=json.loads(str(z["meta"])),
            )

    @property
    def n_users(self) -> int:
        return self.user_emb.shape[0]

    @property
    def n_topics(self) -> int:
        return self.topic_emb.shape[0]

    def topic_label(self, topic: int) -> str:
        """Readable name of a knowledge component."""
        name = str(self.topic_name[topic])
        return name[:1].upper() + name[1:]

    def related(self, topic: int) -> list[str]:
        """Names of the knowledge components DBE-KT22 links to this one."""
        kc_to_topic = {int(k): i for i, k in enumerate(self.topic_kc)}
        ids = [int(x) for x in str(self.topic_related[topic]).split(";") if x]
        return [self.topic_label(kc_to_topic[k]) for k in ids if k in kc_to_topic]

    def time_term(self, dt: np.ndarray) -> np.ndarray:
        """Learned time term for an array of days since practice (NaN = never practised)."""
        if not self.use_forgetting:
            return np.zeros_like(dt, dtype=np.float32)
        seen = ~np.isnan(dt)
        h = np.broadcast_to(self.half_life, dt.shape)
        decay = 1 - np.power(2.0, -np.nan_to_num(dt) / h)
        return np.where(seen, self.beta * decay, self.unseen).astype(np.float32)

    def score_matrix(self) -> np.ndarray:
        """Scores of every (student, topic) at export time; used to check the export against PyTorch."""
        return self.user_emb @ self.topic_emb.T + self.time_term(self.practice_dt)

    def recommend(
        self,
        user: int,
        k: int = 10,
        days_ahead: float = 0.0,
        practised_only: bool = False,
        hidden: set[int] | frozenset[int] = frozenset(),
    ) -> list[dict]:
        """Top-k knowledge components to revise for one student, with the evidence behind each one.

        Args:
            user: Student index (0 to n_users - 1).
            k: Number of topics returned.
            days_ahead: Shift every "days since practice" by this many days (simulates waiting).
            practised_only: Rank only topics the student has already answered.
            hidden: Topics the student dismissed in the app: removed from the list.

        Returns:
            One dict per topic: rank, topic, label, description, score, priority (percentile among
            all topics), graph and time parts of the score, evidence and a template explanation.
        """
        dt = self.practice_dt[user]
        dt = np.where(np.isnan(dt), np.nan, dt + days_ahead)

        graph = self.topic_emb @ self.user_emb[user]
        time = self.time_term(dt)
        score = graph + time
        # Priority = share of the topics ranked below this one (0-100), easier to read than a raw score.
        priority = 100.0 * (np.argsort(np.argsort(score)) / max(self.n_topics - 1, 1))

        allowed = np.ones(self.n_topics, bool)
        if practised_only:
            allowed &= self.attempts[user] > 0
        allowed[list(hidden)] = False
        order = [t for t in np.argsort(-score, kind="stable") if allowed[t]][:k]

        # Typical graph score of this student, to say whether a topic stands out on the graph side.
        graph_median = float(np.median(graph))
        out = []
        for rank, t in enumerate(order, start=1):
            t = int(t)
            out.append({
                "rank": rank,
                "topic": t,
                "label": self.topic_label(t),
                "description": str(self.topic_description[t]),
                "related": self.related(t),
                "score": round(float(score[t]), 4),
                "priority": round(float(priority[t]), 1),
                "graph_score": round(float(graph[t]), 4),
                "time_score": round(float(time[t]), 4),
                "attempts": int(self.attempts[user, t]),
                "errors": int(self.fails[user, t]),
                "days_since_practice": None if np.isnan(dt[t]) else round(float(dt[t]), 1),
                "reason": self.explain(user, t, dt[t], graph[t] - graph_median),
            })
        return out

    def explain(self, user: int, topic: int, dt: float, graph_vs_median: float) -> str:
        """Template explanation built only from the numbers the model uses."""
        a, f = int(self.attempts[user, topic]), int(self.fails[user, topic])
        parts = []
        if a == 0:
            parts.append("Not practised yet: the model gives new topics a learned prior risk")
        else:
            parts.append(f"{f} error{'s' * (f != 1)} in {a} answer{'s' * (a != 1)} ({100 * f / a:.0f}%)")
        if graph_vs_median > 0:
            parts.append("students who fail the same topics as you often fail this one too")
        if self.use_forgetting and not np.isnan(dt):
            if self.beta > 0:
                h = float(self.half_life[topic])
                parts.append(f"last practised {dt:.1f} days ago, about {100 * (1 - 2 ** (-dt / h)):.0f}% forgotten "
                             f"(half-life {h:.1f} days)")
            elif dt < 1:
                parts.append("practised in the last day: topics you are working on now are the ones failed next")
            else:
                parts.append(f"last practised {dt:.0f} days ago")
        return "; ".join(parts) + "."

    def profile(self, user: int) -> dict:
        """Summary of a student's history: answers, error rate, errors per difficulty, weakest topics."""
        att, fai = self.attempts[user], self.fails[user]
        by_level = []
        for i, name in LEVELS.items():
            a, f = int(self.level_attempts[user, i - 1]), int(self.level_fails[user, i - 1])
            by_level.append({"level": name, "answers": a, "errors": f, "error_rate": round(f / a, 3) if a else None})
        rate = np.where(att >= 3, fai / np.maximum(att, 1), -1.0)
        weakest = [{"topic": int(t), "label": self.topic_label(int(t)), "errors": int(fai[t]), "attempts": int(att[t])}
                   for t in np.argsort(-rate, kind="stable")[:5] if rate[t] > 0]
        return {
            "student_id": user,
            "dataset_student": int(self.user_id[user]),
            "answers": int(self.n_answers[user]),
            "topics_practised": int((att > 0).sum()),
            "error_rate": round(float(self.level_fails[user].sum() / max(self.level_attempts[user].sum(), 1)), 3),
            "by_level": by_level,
            "weakest_topics": weakest,
        }
