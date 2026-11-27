"""Export a trained model to the NumPy bundle the API serves (no PyTorch at serving time).

The graph is rebuilt from every answer of every student, as of their last answer,
and propagated once with the trained embeddings. The bundle stores the final
student and topic embeddings, the topic bias and prior, and the forgetting
parameters: model outputs only, no answer records.

Usage:
    python -m src.export --model artifacts/dbe/model_full.pt   ->  artifacts/dbe/serving.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.build_graph import build_history
from src.config import ARTIFACTS, EVENTS_FILE, SERVING_FILE, TOPICS_FILE
from src.evaluate_test import load_model
from src.model import GraphTensors

def export(model_path: Path, out: Path = SERVING_FILE) -> dict:
    events = pd.read_parquet(EVENTS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, len(pd.read_parquet(TOPICS_FILE))
    tag, model = load_model(model_path, n_users, n_topics)
    hist = build_history(events, n_users, n_topics)
    with torch.no_grad():
        u_emb, t_emb, _ = model.propagate(GraphTensors.from_cut(hist))
        half_life = model.half_life().numpy()
    u_emb, t_emb = u_emb.numpy(), t_emb.numpy()

    user_ids = events.groupby("user")["user_id"].first().reindex(range(n_users)).to_numpy(np.int64)
    ckpt = torch.load(model_path, map_location="cpu")
    meta = {"model": tag, "trained_epoch": int(ckpt.get("epoch", -1)), "config": ckpt["config"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Model outputs only: no answer records, so the bundle can be shared without the dataset.
    # Each student's history is rebuilt from the local data when the API starts (src.predict).
    np.savez_compressed(
        out, user_ids=user_ids, user_emb=u_emb.astype(np.float32), topic_emb=t_emb.astype(np.float32),
        topic_bias=model.topic_bias.detach().numpy().astype(np.float32),
        topic_prior=GraphTensors.from_cut(hist).topic_prior.numpy().astype(np.float32),
        prior_weight=np.float32(model.prior_weight.item()),
        half_life=half_life.astype(np.float32), beta=np.float32(model.beta.item()), unseen=np.float32(model.unseen.item()),
        use_forgetting=np.bool_(model.use_forgetting), meta=np.array(json.dumps(meta)),
    )
    return {"file": str(out), "kb": round(out.stat().st_size / 1024), **meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "model_full.pt")
    parser.add_argument("--out", type=Path, default=SERVING_FILE)
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.out), indent=2, default=str))


if __name__ == "__main__":
    main()
