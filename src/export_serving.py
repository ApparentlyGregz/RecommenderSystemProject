"""Export the trained model into a small file the API can serve without PyTorch.

The serving "cut" is now: for every student, the inputs are ALL their answers
(the cut sits 1 ms after their last answer). The trained embeddings are
propagated once on that graph, with the learned forgetting weights on the
edges. The API then only needs a dot product and the forgetting term, which it
recomputes at every request (so it can shift time or take feedback into
account).

What is saved in artifacts/serving.npz (under 1 MB, committed to Git):
    user_emb, topic_emb       propagated LightGCN embeddings (float32)
    half_life, beta, unseen   learned forgetting parameters
    practice_dt               days since each (student, topic) was last practised, NaN if never
    attempts, fails           answers and errors per (student, topic)
    user_id, n_answers        DBE-KT22 student id and number of answers per student
    level_attempts/_fails     answers and errors per question difficulty (1 easy, 2 medium, 3 hard)
    topic_kc, topic_name,     DBE-KT22 knowledge component id, name, description and related KCs
    topic_description, topic_related
    meta                      JSON string: checkpoint, epoch, validation NDCG@10, date

Usage:
    python -m src.export_serving                         # artifacts/model_full.pt
    python -m src.export_serving --model artifacts/model_cl_noforget.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.build_graph import history_inputs
from src.config import ARTIFACTS, EVENTS_FILE, TOPICS_FILE
from src.evaluate_test import load_model
from src.model import GraphTensors

SERVING_FILE = ARTIFACTS / "serving.npz"


def serving_graph(events: pd.DataFrame, n_users: int, n_topics: int) -> tuple[GraphTensors, dict[str, np.ndarray]]:
    """Inputs at "now": every answer of every student is history.

    Args:
        events: Processed event table.
        n_users: Number of students.
        n_topics: Number of topics.

    Returns:
        The graph tensors and the raw numpy inputs (attempts, fails, practice_dt...).
    """
    # The cut is 1 ms after the student's last answer, so nothing is left out.
    t_now = events.groupby("user")["timestamp"].max() + 1
    inputs = history_inputs(events, t_now, n_users, n_topics)
    g = GraphTensors(
        n_users=n_users,
        n_topics=n_topics,
        edge_user=torch.as_tensor(inputs["edge_user"]),
        edge_topic=torch.as_tensor(inputs["edge_topic"]),
        edge_dt=torch.as_tensor(inputs["edge_dt"]),
        practice_dt=torch.as_tensor(inputs["practice_dt"]),
    )
    return g, inputs


def main() -> None:
    """Propagate the trained model on the full history and save artifacts/serving.npz."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "model_full.pt")
    parser.add_argument("--out", type=Path, default=SERVING_FILE)
    args = parser.parse_args()

    events = pd.read_parquet(EVENTS_FILE)
    topics = pd.read_parquet(TOPICS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, len(topics)
    tag, model = load_model(args.model, n_users, n_topics)
    g, inputs = serving_graph(events, n_users, n_topics)

    with torch.no_grad():
        u_emb, t_emb, _ = model.propagate(g)
        reference = model.score_all(g).numpy()  # used below to check the export
        half_life = model.half_life().numpy()

    # One row per answer (an answer with two tags appears twice in events).
    answers = events.drop_duplicates(["user", "timestamp", "question"])
    per_user = answers.groupby("user").agg(user_id=("user_id", "first"), n_answers=("question", "size"))
    levels = np.zeros((n_users, 3), np.int64), np.zeros((n_users, 3), np.int64)
    by_level = answers[answers["difficulty"].between(1, 3)].groupby(["user", "difficulty"])["correct"].agg(["size", "sum"])
    for (u, d), row in by_level.iterrows():
        levels[0][u, d - 1] = row["size"]
        levels[1][u, d - 1] = row["size"] - row["sum"]

    ckpt = torch.load(args.model, map_location="cpu")
    history_file = ARTIFACTS / f"history_{tag}.json"
    best_val = None
    if history_file.exists():
        best_val = max(r["val_ndcg@10"] for r in json.loads(history_file.read_text()))
    meta = {
        "model": tag,
        "checkpoint": args.model.name,
        "epoch": ckpt.get("epoch"),
        "best_val_ndcg@10": best_val,
        "use_forgetting": model.use_forgetting,
        "use_contrastive": model.use_contrastive,
        "exported": time.strftime("%Y-%m-%d %H:%M"),
    }
    arrays = {
        "user_emb": u_emb.numpy().astype(np.float32),
        "topic_emb": t_emb.numpy().astype(np.float32),
        "half_life": half_life.astype(np.float32),
        "beta": np.float32(model.beta.item() if model.use_forgetting else 0.0),
        "unseen": np.float32(model.unseen.item() if model.use_forgetting else 0.0),
        "use_forgetting": np.bool_(model.use_forgetting),
        "practice_dt": inputs["practice_dt"].astype(np.float32),
        "attempts": inputs["attempts"].astype(np.uint16),
        "fails": inputs["fails"].astype(np.uint16),
        "user_id": per_user["user_id"].to_numpy(np.int64),
        "n_answers": per_user["n_answers"].to_numpy(np.int32),
        "level_attempts": levels[0].astype(np.uint16),
        "level_fails": levels[1].astype(np.uint16),
        "topic_kc": topics["kc"].to_numpy(np.int32),
        "topic_name": topics["name"].to_numpy(str),
        "topic_description": topics["description"].to_numpy(str),
        "topic_related": topics["related_kcs"].to_numpy(str),
        "meta": np.array(json.dumps(meta)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    # The numpy predictor must give exactly the scores of the PyTorch model.
    from src.predict import Predictor

    pred = Predictor.load(args.out)
    gap = float(np.nanmax(np.abs(pred.score_matrix() - reference)))
    print(json.dumps({**meta, "file": str(args.out), "mb": round(args.out.stat().st_size / 1e6, 2),
                      "max_abs_gap_vs_torch": gap}, indent=2))
    assert gap < 1e-4, "the exported predictor does not match the PyTorch model"


if __name__ == "__main__":
    main()
