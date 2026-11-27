"""DBE-KT22 -> the event table of the pipeline, plus the questions shown in the app.

DBE-KT22 (Abdelrahman et al., 2022) records students of a database course at the
Australian National University answering multiple-choice questions, with the
question text, the choices and the knowledge components (KCs, our "topics").

Steps:
    1. Keep students with at least MIN_ANSWERS answers (3 label windows + history).
    2. One row per (answer, KC); answers ordered by start time; answer_idx counts from the end.
    3. Write data/processed/dbe/: events.parquet, topics.parquet (KC names and
       descriptions), questions.parquet (text split into HTML and LaTeX blocks,
       choices, hint, explanation, KCs) and stats.json.

Usage:
    python -m src.prepare_dbe --raw data/raw/dbe/2_DBE_KT22_datafiles_100102_csv.zip
    (a folder with the eight CSV files works too)
"""
from __future__ import annotations

import argparse
import html
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, DATASET, EVENTS_FILE, MIN_ANSWERS, QUESTIONS_FILE, STATS_FILE, TOPICS_FILE

DEFAULT_RAW = DATA_RAW / "dbe" / "2_DBE_KT22_datafiles_100102_csv.zip"
# Relations of the course worksheets. Some questions use them without showing their rows,
# which DBE-KT22 does not include: those questions are not self-contained.
WORKSHEET_RELATIONS = re.compile(r"\b(?:Study|Enrol|Course|Movie|Writer|Person|Director|Student|Employee)\b")
LATEX_IMG = re.compile(r'<img[^>]*src="https?://latex\.codecogs\.com/gif\.latex\?([^"]+)"[^>]*/?>', re.I)
STYLE_WRAPPER = re.compile(r"\\(?:bf|large|color\{[a-z]+\})\s*")


def read_tables(raw: Path) -> dict[str, pd.DataFrame]:
    """Read the CSV files of DBE-KT22 from the zip or a folder, keyed by lower-case name."""
    out = {}
    if raw.suffix == ".zip":
        with zipfile.ZipFile(raw) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    out[Path(name).stem.lower()] = pd.read_csv(io.BytesIO(zf.read(name)))
    else:
        for path in raw.glob("*.csv"):
            out[path.stem.lower()] = pd.read_csv(path)
    missing = {"transaction", "questions", "question_choices", "kcs", "question_kc_relationships"} - set(out)
    if missing:
        raise FileNotFoundError(f"DBE-KT22 files missing in {raw}: {sorted(missing)}")
    return out


def clean_latex(code: str) -> str:
    """LaTeX of a codecogs image without its display wrappers (bold, large, red)."""
    code = STYLE_WRAPPER.sub("", code.replace("\r", " ").replace("\n", " ")).strip()
    while code.startswith("{") and code.endswith("}") and code.count("{") == code.count("}"):
        inner = code[1:-1].strip()
        if inner.count("{") != inner.count("}"):
            break
        code = inner
    return code


def question_blocks(rich: str) -> list[dict]:
    """Split a question's HTML into blocks the app can render without external images.

    Short formulas (like T_1) become inline HTML; long ones (relational algebra)
    become their own LaTeX block, rendered by the interface.
    """
    rich = re.sub(r"(?is)<(head|style|script)[^>]*>.*?</\1>", "", rich or "")
    rich = re.sub(r"(?i)</?(html|body|meta)[^>]*>", "", rich)
    blocks, last = [], 0
    for m in LATEX_IMG.finditer(rich):
        code = clean_latex(html.unescape(m.group(1)))
        simple = re.fullmatch(r"\s*([A-Za-z])_\{?(\w+)\}?\s*", code)
        if simple:  # inline subscript, e.g. T_{1}
            rich = rich[: m.start()] + f"{simple.group(1)}<sub>{simple.group(2)}</sub>" + rich[m.end():]
            return question_blocks(rich)
        blocks.append({"type": "html", "content": rich[last:m.start()]})
        blocks.append({"type": "latex", "content": code})
        last = m.end()
    blocks.append({"type": "html", "content": rich[last:]})
    return [b for b in blocks if b["content"].strip() and re.sub(r"<br\s*/?>|\s", "", b["content"])]


def build(raw: Path) -> dict:
    t = read_tables(raw)
    tx, q, ch, kcs, qkc = t["transaction"], t["questions"], t["question_choices"], t["kcs"], t["question_kc_relationships"]

    start = pd.to_datetime(tx["start_time"], utc=True)
    epoch_ms = (start - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)  # independent of the dtype resolution
    tx = tx.assign(timestamp=epoch_ms.astype(np.int64),
                   correct=tx["answer_state"].astype(str).str.lower().eq("true").astype(np.int8))
    per_student = tx.groupby("student_id").size()
    keep = per_student[per_student >= MIN_ANSWERS].index
    tx = tx[tx["student_id"].isin(keep)].sort_values(["student_id", "timestamp", "id"], kind="stable")
    tx["answer_idx"] = tx.groupby("student_id").cumcount(ascending=False).astype(np.int32)

    used_kcs = sorted(qkc["knowledgecomponent_id"].unique())
    topics = kcs.set_index("id").loc[used_kcs].reset_index().rename(columns={"id": "kc_id"})
    topics["name"] = topics["name"].str.strip()
    topics["description"] = topics["description"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    topics["topic"] = np.arange(len(topics), dtype=np.int16)
    kc_to_topic = dict(zip(topics["kc_id"], topics["topic"]))

    ev = tx.merge(qkc[["question_id", "knowledgecomponent_id"]], on="question_id")
    ev["topic"] = ev["knowledgecomponent_id"].map(kc_to_topic).astype(np.int16)
    users = np.sort(ev["student_id"].unique())
    ev["user_id"] = ev["student_id"].astype(np.int64)
    ev["user"] = ev["student_id"].map({u: i for i, u in enumerate(users)}).astype(np.int32)
    ev["question"] = ev["question_id"].astype(np.int32)
    ev["part"] = np.int8(0)  # EdNet column (TOEIC part), unused here
    events = ev[["user", "user_id", "timestamp", "answer_idx", "question", "part", "topic", "correct"]]
    events = events.sort_values(["user", "timestamp", "question", "topic"]).reset_index(drop=True)
    counts = events.groupby("topic").size()
    topics["n_events"] = topics["topic"].map(counts).fillna(0).astype(int)
    topics["n_questions"] = topics["kc_id"].map(qkc.groupby("knowledgecomponent_id")["question_id"].nunique()).astype(int)

    choices = ch.assign(is_correct=ch["is_correct"].astype(str).str.lower().eq("true"))
    choices = choices.sort_values(["question_id", "id"])
    by_q_choices = {qid: [{"id": int(r.id), "text": str(r.choice_text).strip(), "is_correct": bool(r.is_correct)}
                          for r in g.itertuples()] for qid, g in choices.groupby("question_id")}
    by_q_topics = qkc.groupby("question_id")["knowledgecomponent_id"].apply(lambda s: [kc_to_topic[k] for k in s])
    qs = pd.DataFrame({
        "question": q["id"].astype(np.int32),
        "title": q["question_title"].fillna("").astype(str),
        "blocks": [json.dumps(question_blocks(r)) for r in q["question_rich_text"].fillna("")],
        "text": q["question_text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "hint": q["hint_text"].fillna("").astype(str).str.strip(),
        "explanation": q["explanation"].fillna("").astype(str).str.strip(),
        "difficulty": q["difficulty"].fillna(0).astype(int),
        "choices": [json.dumps(by_q_choices.get(i, [])) for i in q["id"]],
        "topics": [json.dumps([int(x) for x in by_q_topics.get(i, [])]) for i in q["id"]],
    }).sort_values("question").reset_index(drop=True)
    full_text = qs["text"] + " " + qs["blocks"]
    qs["self_contained"] = ~(full_text.str.contains(WORKSHEET_RELATIONS) & ~qs["blocks"].str.contains("<table", case=False))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS_FILE, index=False)
    topics[["topic", "kc_id", "name", "description", "n_events", "n_questions"]].to_parquet(TOPICS_FILE, index=False)
    qs.to_parquet(QUESTIONS_FILE, index=False)
    first = events.drop_duplicates(["user", "timestamp", "question"])
    return {
        "students": int(len(users)),
        "students_before_filter": int(len(per_student)),
        "answers": int(len(first)),
        "events_answer_x_topic": int(len(events)),
        "topics": int(len(topics)),
        "questions": int(len(qs)),
        "questions_self_contained": int(qs["self_contained"].sum()),
        "answers_per_student_median": float(first.groupby("user").size().median()),
        "overall_error_rate": float(1 - first["correct"].mean()),
        "date_range": [str(pd.to_datetime(events["timestamp"].min(), unit="ms").date()),
                       str(pd.to_datetime(events["timestamp"].max(), unit="ms").date())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="DBE-KT22 csv zip or folder")
    args = parser.parse_args()
    if DATASET != "dbe":
        raise SystemExit("Set REVISE_MALIN_DATASET=dbe (the default) to prepare DBE-KT22.")
    start = time.time()
    stats = build(args.raw)

    from src.build_graph import load_cuts, window_stats  # after the files exist

    events = pd.read_parquet(EVENTS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, stats["topics"]
    stats["cuts"] = {name: window_stats(c) for name, c in load_cuts(events, n_users, n_topics).items()}
    stats["seconds"] = round(time.time() - start, 1)
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
