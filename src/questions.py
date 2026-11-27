"""DBE-KT22 questions for the practice mode of the app: text, choices, hint, explanation, topics.

The text is split into blocks the interface can render without external images:
HTML as it is, and the relational algebra that DBE-KT22 draws with codecogs
images as LaTeX. Some questions use tables from the course worksheet that the
dataset does not include; they are flagged (self_contained = False) and the app
does not offer them, since they cannot be answered without those tables.

The question text stays local (data/processed/questions.parquet, not in Git):
it is built from the raw DBE-KT22 files, which `run.py` downloads on first use.

Usage:
    python -m src.questions                          # reads data/raw/dbe-kt22/
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, QUESTIONS_FILE, TOPICS_FILE

RAW_DIR = DATA_RAW / "dbe-kt22"
LATEX_IMG = re.compile(r'<img[^>]*src="https?://latex\.codecogs\.com/gif\.latex\?([^"]+)"[^>]*/?>', re.I)
STYLE_WRAPPER = re.compile(r"\\(?:bf|large|color\{[a-z]+\})\s*")
# Relations of the course worksheets: questions that use them without showing their rows are not self-contained.
WORKSHEET_RELATIONS = re.compile(r"\b(?:Study|Enrol|Course|Movie|Writer|Person|Director|Student|Employee)\b")


def clean_latex(code: str) -> str:
    """LaTeX of a codecogs image without its display wrappers (bold, large, red)."""
    code = STYLE_WRAPPER.sub("", code.replace("\r", " ").replace("\n", " ")).strip()
    while code.startswith("{") and code.endswith("}"):
        inner = code[1:-1].strip()
        if inner.count("{") != inner.count("}"):
            break
        code = inner
    return code


def question_blocks(rich: str) -> list[dict]:
    """Split a question's HTML into {"type": "html" | "latex", "content"} blocks.

    Short formulas such as T_1 become inline HTML subscripts; longer ones become LaTeX blocks.
    """
    rich = re.sub(r"(?is)<(head|style|script)[^>]*>.*?</\1>", "", rich or "")
    rich = re.sub(r"(?i)</?(html|body|meta)[^>]*>", "", rich)
    blocks, last = [], 0
    for m in LATEX_IMG.finditer(rich):
        code = clean_latex(html.unescape(m.group(1)))
        simple = re.fullmatch(r"\s*([A-Za-z])_\{?(\w+)\}?\s*", code)
        if simple:  # inline subscript, then parse the rest again
            return question_blocks(rich[: m.start()] + f"{simple.group(1)}<sub>{simple.group(2)}</sub>" + rich[m.end():])
        blocks.append({"type": "html", "content": rich[last:m.start()]})
        blocks.append({"type": "latex", "content": code})
        last = m.end()
    blocks.append({"type": "html", "content": rich[last:]})
    return [b for b in blocks if re.sub(r"<br\s*/?>|\s", "", b["content"])]


def build(raw: Path = RAW_DIR, topics_file: Path = TOPICS_FILE) -> pd.DataFrame:
    """Questions table indexed like the model's topics (topics.parquet maps DBE-KT22 KC ids to topics)."""
    q = pd.read_csv(raw / "Questions.csv")
    ch = pd.read_csv(raw / "Question_Choices.csv")
    qkc = pd.read_csv(raw / "Question_KC_Relationships.csv")
    topics = pd.read_parquet(topics_file)
    kc_to_topic = dict(zip(topics["kc"], topics["topic"]))

    ch = ch.assign(is_correct=ch["is_correct"].astype(str).str.lower().eq("true")).sort_values(["question_id", "id"])
    choices = {qid: [{"id": int(r.id), "text": str(r.choice_text).strip(), "is_correct": bool(r.is_correct)}
                     for r in g.itertuples()] for qid, g in ch.groupby("question_id")}
    by_q = qkc.groupby("question_id")["knowledgecomponent_id"].apply(
        lambda s: sorted({int(kc_to_topic[k]) for k in s if k in kc_to_topic}))
    out = pd.DataFrame({
        "question": q["id"].astype(np.int32),
        "title": q["question_title"].fillna("").astype(str),
        "blocks": [json.dumps(question_blocks(r)) for r in q["question_rich_text"].fillna("")],
        "text": q["question_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "hint": q["hint_text"].fillna("").astype(str).str.strip(),
        "explanation": q["explanation"].fillna("").astype(str).str.strip(),
        "difficulty": q["difficulty"].fillna(0).astype(int),
        "choices": [json.dumps(choices.get(i, [])) for i in q["id"]],
        "topics": [json.dumps(by_q.get(i, [])) for i in q["id"]],
    }).sort_values("question").reset_index(drop=True)
    uses_worksheet = (out["text"] + " " + out["blocks"]).str.contains(WORKSHEET_RELATIONS)
    out["self_contained"] = ~(uses_worksheet & ~out["blocks"].str.contains("<table", case=False))
    return out[out["choices"] != "[]"].reset_index(drop=True)


class QuestionBank:
    """Practice questions per topic, served without their answers; answers are checked here."""

    def __init__(self, table: pd.DataFrame, topic_names: list[str]):
        self.names = topic_names
        self.q = table.assign(
            choices_list=table["choices"].map(json.loads),
            topics_list=table["topics"].map(json.loads),
            blocks_list=table["blocks"].map(json.loads),
        ).set_index("question", drop=False).rename_axis("qid")

    @classmethod
    def load(cls, path: Path, topic_names: list[str]) -> "QuestionBank":
        return cls(pd.read_parquet(path), topic_names)

    def _pool(self, topic: int, exclude: set[int], include_context: bool) -> pd.DataFrame:
        pool = self.q[self.q["topics_list"].map(lambda ts: topic in ts) & ~self.q["question"].isin(exclude)]
        return pool if include_context else pool[pool["self_contained"]]

    def count_for_topic(self, topic: int, exclude: set[int] = frozenset()) -> int:
        """Number of self-contained questions of a topic left to practise (worksheet questions not counted)."""
        return len(self._pool(topic, exclude, include_context=False))

    def for_topic(self, topic: int, n: int = 1, exclude: set[int] = frozenset(),
                  include_context: bool = False) -> list[dict]:
        """Questions of a topic, easier ones first; no correct answer included.

        Questions that need the course worksheet tables (not in DBE-KT22) cannot be answered,
        so they are left out unless include_context is True; they then come last.
        """
        pool = self._pool(topic, exclude, include_context)
        pool = pool.assign(needs_context=~pool["self_contained"]).sort_values(["needs_context", "difficulty", "question"])
        return [{
            "question": int(r.question), "title": r.title, "blocks": r.blocks_list, "difficulty": int(r.difficulty),
            "hint": r.hint, "needs_context": bool(r.needs_context),
            "choices": [{"id": c["id"], "text": c["text"]} for c in r.choices_list],
            "topics": [{"topic": t, "label": self.names[t]} for t in r.topics_list],
        } for r in pool.head(n).itertuples()]

    def check(self, question: int, choice: int) -> dict:
        """Whether a choice is correct, with the correct choices, the explanation and the question's topics."""
        if int(question) not in self.q.index:
            raise KeyError(f"Unknown question {question}")
        r = self.q.loc[int(question)]
        if int(choice) not in {c["id"] for c in r["choices_list"]}:
            raise KeyError(f"Choice {choice} does not belong to question {question}")
        good = [c["id"] for c in r["choices_list"] if c["is_correct"]]
        return {"question": int(question), "choice": int(choice), "correct": int(choice) in good,
                "correct_choices": good, "explanation": r["explanation"], "topics": list(r["topics_list"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="folder with the DBE-KT22 CSV files")
    args = parser.parse_args()
    qs = build(args.raw)
    QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    qs.to_parquet(QUESTIONS_FILE, index=False)
    print(f"{len(qs)} questions ({int(qs['self_contained'].sum())} self-contained) -> {QUESTIONS_FILE}")


if __name__ == "__main__":
    main()
