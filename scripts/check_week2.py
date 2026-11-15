"""Print the evidence for the Week 2 risk questions Q1-Q4 and save it to reports/.

Run after src.prepare_data:
    python scripts/check_week2.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.build_graph import load_cuts  # noqa: E402
from src.config import EVENTS_FILE, STATS_FILE, TOPICS_FILE  # noqa: E402
from src.evaluate import evaluate_scores, run_baselines  # noqa: E402


def q1_data() -> None:
    stats = json.loads(STATS_FILE.read_text())
    print("Q1 DATA - processed sample (data/processed/stats.json)")
    for key in ["students", "answers", "events_answer_x_topic", "topics", "questions_seen",
                "answers_per_student_median", "overall_error_rate", "parquet_mb", "seconds"]:
        print(f"  {key}: {stats[key]}")
    print("  test window:", json.dumps(stats["cuts"]["test"]))


def q2_model(events: pd.DataFrame, n_users: int, n_topics: int) -> None:
    import torch

    from src.model import GraphTensors, ReviseMalin

    print("\nQ2 MODEL - one forward pass of NLGCL + forgetting on the train cut")
    torch.manual_seed(0)
    cut = load_cuts(events, n_users, n_topics)["train"]
    g = GraphTensors.from_cut(cut)
    model = ReviseMalin(n_users, n_topics)
    start = time.time()
    u_emb, t_emb, layers = model.propagate(g)
    users = torch.as_tensor(cut.cand_user[:4096])
    topics = torch.as_tensor(cut.cand_topic[:4096])
    out = model.loss(g, users, topics, torch.roll(topics, 1))
    out["total"].backward()
    grads_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"  graph: {n_users} students, {n_topics} topics, {len(cut.edge_user):,} edges")
    print(f"  embeddings: students {tuple(u_emb.shape)}, topics {tuple(t_emb.shape)}, layers {len(layers)}")
    print("  loss:", {k: round(float(v.detach()), 4) for k, v in out.items()})
    print(f"  gradients finite: {grads_ok}; forward + backward in {time.time() - start:.2f}s")


def q3_evaluation(events: pd.DataFrame, n_users: int, n_topics: int) -> None:
    print("\nQ3 EVALUATION - harness on mock predictions, then baselines (test cut)")
    cut = load_cuts(events, n_users, n_topics)["test"]
    mock = np.random.default_rng(123).normal(size=(n_users, n_topics))
    res = evaluate_scores(mock, cut)
    print("  mock predictions:", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in res.items() if k != "per_student"})
    table = run_baselines(cut)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def q4_system() -> None:
    from fastapi.testclient import TestClient

    print("\nQ4 SYSTEM - FastAPI returns mock recommendations and logs a click in SQLite")
    os.environ["REVISE_MALIN_DB"] = str(Path(tempfile.mkdtemp()) / "check.db")
    from api.main import app

    with TestClient(app) as c:
        rec = c.get("/recommend", params={"student_id": 1, "k": 3}).json()
        log = c.post("/log_interaction", json={"student_id": 1, "topic": 52, "action": "revised"}).json()
        stored = c.get("/interactions/1").json()
    print("  GET /recommend?student_id=1&k=3 ->", json.dumps(rec))
    print("  POST /log_interaction ->", json.dumps(log))
    print(f"  rows in SQLite for student 1: {len(stored)}")


def main() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip() or "no git"
        print(f"Week 2 evidence - commit {commit} - {time.strftime('%Y-%m-%d %H:%M')}\n")
        events = pd.read_parquet(EVENTS_FILE)
        n_users, n_topics = int(events["user"].max()) + 1, len(pd.read_parquet(TOPICS_FILE))
        q1_data()
        q2_model(events, n_users, n_topics)
        q3_evaluation(events, n_users, n_topics)
        q4_system()
    text = buf.getvalue()
    print(text)
    out = ROOT / "reports" / "week2_evidence.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
