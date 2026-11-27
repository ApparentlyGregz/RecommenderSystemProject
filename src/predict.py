"""Serve recommendations with NumPy only (no PyTorch at serving time).

Loads the bundle written by src.export plus the topic and question tables, and
answers three questions for the app:

- recommend(): which skills should this student revise first, and why?
- questions(): real questions of a skill to practise now.
- check(): is the chosen answer correct, and what is the explanation?

Ranking. The trained model scores every (student, skill) with the propagated
embeddings, the skill bias and the forgetting term if the model has one. Answers
given in the app then update the ranking with a simple, documented rule (not
learned): the model's rank counts as two pseudo-answers, and each answer in the
app moves it up (wrong) or down (correct). A skill practised in the app also has
a time gap of zero.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EVENTS_FILE, MS_PER_DAY, QUESTIONS_FILE, SERVING_FILE, TOPICS_FILE

PRIOR_ANSWERS = 2.0  # weight of the model rank against answers given in the app

MODEL_NAMES = {
    "full": "NLGCL + forgetting",
    "cl_noforget": "NLGCL",
    "nocl_forget": "LightGCN + forgetting",
    "nocl_noforget": "LightGCN",
}


N_NEIGHBOURS = 20


def history_arrays(events: pd.DataFrame, bundle: dict, n_topics: int) -> dict:
    """Per-student history the explanations need, from the event table (pandas and NumPy only).

    Returns attempts, fails, last_ts (per student and topic), now (last answer + 1 ms),
    answered (student x question) with its question ids, and the 20 nearest students
    by embedding (used for "students with answers like yours").
    """
    user_ids = bundle["user_ids"]
    events = events[events["user_id"].isin(user_ids)]
    row = {int(s): i for i, s in enumerate(user_ids)}
    u_idx = events["user_id"].map(row).to_numpy()
    n_users = len(user_ids)
    attempts = np.zeros((n_users, n_topics), np.int16)
    fails = np.zeros((n_users, n_topics), np.int16)
    np.add.at(attempts, (u_idx, events["topic"].to_numpy()), 1)
    np.add.at(fails, (u_idx, events["topic"].to_numpy()), 1 - events["correct"].to_numpy())
    last_ts = np.full((n_users, n_topics), np.nan)
    last = events.assign(u=u_idx).groupby(["u", "topic"])["timestamp"].max().reset_index()
    last_ts[last["u"], last["topic"]] = last["timestamp"]
    now = events.assign(u=u_idx).groupby("u")["timestamp"].max().reindex(range(n_users)).to_numpy(np.int64) + 1
    first = events.drop_duplicates(["user_id", "timestamp", "question"])
    questions = np.sort(first["question"].unique())
    answered = np.zeros((n_users, len(questions)), dtype=bool)
    answered[first["user_id"].map(row).to_numpy(), np.searchsorted(questions, first["question"].to_numpy())] = True
    emb = bundle["user_emb"]
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    neighbours = np.argsort(-sim, axis=1)[:, :N_NEIGHBOURS]
    return {"attempts": attempts, "fails": fails, "last_ts": last_ts, "now": now, "questions": questions,
            "answered": answered, "neighbours": neighbours}


@dataclass
class SessionAnswer:
    """One answer given in the app."""

    question: int
    topics: list[int]
    correct: bool


@dataclass
class Recommender:
    """Ranks skills for one student from the exported bundle."""

    bundle: dict
    topics: pd.DataFrame
    questions: pd.DataFrame
    _index: dict = field(init=False)

    def __post_init__(self) -> None:
        self._index = {int(s): i for i, s in enumerate(self.bundle["user_ids"])}
        self.meta = json.loads(str(self.bundle["meta"]))
        cfg = self.meta.get("config", {})
        name = MODEL_NAMES.get(self.meta["model"], self.meta["model"])
        self.meta["name"] = name + (" with skill difficulty bias" if cfg.get("topic_bias") else "")
        a, f = self.bundle["attempts"].astype(float), self.bundle["fails"].astype(float)
        self.global_rate = (f.sum(0) + 1) / (a.sum(0) + 2)  # error rate of each skill over all students
        self.questions = self.questions.assign(
            choices_list=self.questions["choices"].map(json.loads),
            topics_list=self.questions["topics"].map(json.loads),
            blocks_list=self.questions["blocks"].map(json.loads),
        ).set_index("question", drop=False).rename_axis("qid")

    @classmethod
    def load(cls, bundle_path: Path = SERVING_FILE, topics_path: Path = TOPICS_FILE,
             questions_path: Path = QUESTIONS_FILE, events_path: Path = EVENTS_FILE) -> "Recommender":
        with np.load(bundle_path, allow_pickle=False) as npz:
            bundle = {k: npz[k] for k in npz.files}
        topics = pd.read_parquet(topics_path)
        if "attempts" not in bundle:  # model-only bundle: rebuild each student's history from the local data
            bundle.update(history_arrays(pd.read_parquet(events_path), bundle, len(topics)))
        return cls(bundle, topics, pd.read_parquet(questions_path))

    # ------------------------------------------------------------------ students
    def has_student(self, student_id: int) -> bool:
        return int(student_id) in self._index

    def students(self) -> list[dict]:
        """Every student of the bundle with their history size and error rate."""
        a, f = self.bundle["attempts"], self.bundle["fails"]
        answered = self.bundle["answered"].sum(1)
        return [{"student_id": int(s), "answers": int(answered[i]),
                 "error_rate": round(float(f[i].sum() / max(a[i].sum(), 1)), 3)}
                for i, s in enumerate(self.bundle["user_ids"])]

    # ------------------------------------------------------------------ ranking
    def _model_scores(self, u: int, practised_now: set[int]) -> np.ndarray:
        b = self.bundle
        dot = b["topic_emb"] @ b["user_emb"][u] + b.get("topic_bias", 0.0)
        if "topic_prior" in b:
            dot = dot + float(b["prior_weight"]) * b["topic_prior"]
        if not bool(b["use_forgetting"]):
            return dot
        dt = (b["now"][u] - b["last_ts"][u]) / MS_PER_DAY
        seen = ~np.isnan(dt)
        for t in practised_now:
            dt[t], seen[t] = 0.0, True
        decay = 1 - np.power(2.0, -np.nan_to_num(dt) / b["half_life"])
        return dot + np.where(seen, float(b["beta"]) * decay, float(b["unseen"]))

    def recommend(self, student_id: int, k: int = 10, session: list[SessionAnswer] | None = None,
                  revised: set[int] = frozenset(), hidden: set[int] = frozenset()) -> list[dict]:
        """Top-k skills to revise, with the evidence behind each one.

        Args:
            student_id: Dataset student id.
            k: Number of skills.
            session: Answers given in the app.
            revised: Skills the student marked as revised (time gap set to zero).
            hidden: Skills marked "already known" or "not useful", left out of the list.
        """
        u = self._index[int(student_id)]
        session = session or []
        n_app = np.zeros(len(self.topics))
        wrong_app = np.zeros(len(self.topics))
        for ans in session:
            for t in ans.topics:
                n_app[t] += 1
                wrong_app[t] += not ans.correct
        scores = self._model_scores(u, {t for t in range(len(n_app)) if n_app[t] > 0} | set(revised))
        model_rank = scores.argsort().argsort() / max(len(scores) - 1, 1)  # 0 = safest, 1 = riskiest
        # Session update: the model rank counts as PRIOR_ANSWERS pseudo-answers, then each answer in
        # the app moves it: a wrong one up, a correct one down (a beta-binomial style average).
        risk = (wrong_app + PRIOR_ANSWERS * model_rank) / (n_app + PRIOR_ANSWERS)
        risk[list(hidden)] = -np.inf
        order = [t for t in np.argsort(-risk, kind="stable") if np.isfinite(risk[t])][:k]
        return [self._explain(u, int(t), float(risk[t]), float(scores[t]), int(n_app[t]), int(wrong_app[t]))
                for t in order]

    def _explain(self, u: int, t: int, risk: float, score: float, n_app: int, wrong_app: int) -> dict:
        b = self.bundle
        att, fail = int(b["attempts"][u, t]), int(b["fails"][u, t])
        nb = b["neighbours"][u]
        nb_att, nb_fail = b["attempts"][nb, t].sum(), b["fails"][nb, t].sum()
        reasons = []
        if att:
            reasons.append(f"You got {fail} of your {att} answers on this skill wrong.")
        else:
            reasons.append("You have not practised this skill yet.")
        if nb_att >= 3:
            reasons.append(f"Students with answers like yours got {nb_fail / nb_att:.0%} wrong on it.")
        dt = (b["now"][u] - b["last_ts"][u, t]) / MS_PER_DAY
        if att and dt >= 1:
            reasons.append(f"Last practised {dt:.0f} days before your last session.")
        if n_app:
            reasons.append(f"In this session: {n_app - wrong_app} of {n_app} correct.")
        row = self.topics.iloc[t]
        return {
            "topic": t, "skill": row["name"], "description": row["description"], "risk": round(risk, 3),
            "model_score": round(score, 4), "attempts": att, "errors": fail,
            "skill_error_rate": round(float(self.global_rate[t]), 3), "reasons": reasons,
            "n_questions": int(row["n_questions"]),
        }

    # ------------------------------------------------------------------ questions
    def questions_for(self, student_id: int, topic: int, n: int = 3, exclude: set[int] = frozenset()) -> list[dict]:
        """Questions of a skill to practise, the ones the student has not answered first; no answers included."""
        u = self._index[int(student_id)]
        done = set(self.bundle["questions"][self.bundle["answered"][u]].tolist())
        pool = self.questions[self.questions["topics_list"].map(lambda ts: topic in ts)]
        pool = pool[~pool["question"].isin(exclude)]
        self_contained = pool["self_contained"] if "self_contained" in pool else True
        pool = pool.assign(seen=pool["question"].isin(done), needs_context=~pd.Series(self_contained, index=pool.index))
        pool = pool.sort_values(["needs_context", "seen", "difficulty", "question"])
        return [self._public(q) for q in pool.head(n).itertuples()]

    def _public(self, q) -> dict:
        return {
            "question": int(q.question), "title": q.title, "blocks": q.blocks_list, "difficulty": int(q.difficulty),
            "needs_context": bool(getattr(q, "needs_context", False)),
            "hint": q.hint, "choices": [{"id": c["id"], "text": c["text"]} for c in q.choices_list],
            "topics": [{"topic": t, "skill": self.topics.iloc[t]["name"]} for t in q.topics_list],
        }

    def check(self, question: int, choice: int) -> dict:
        """Correctness of a chosen answer, with the correct choice and the explanation."""
        q = self.questions.loc[int(question)]
        good = [c["id"] for c in q["choices_list"] if c["is_correct"]]
        if int(choice) not in {c["id"] for c in q["choices_list"]}:
            raise KeyError(f"Choice {choice} does not belong to question {question}")
        return {"question": int(question), "choice": int(choice), "correct": int(choice) in good,
                "correct_choices": good, "explanation": q["explanation"], "topics": list(q["topics_list"])}
