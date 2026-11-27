"""Train Révise Malin with early stopping on validation NDCG@10.

Training pairs come from the train cut only: for each student, a topic
attempted and failed in the label window (positive) against a topic attempted
and always correct in the same window (negative). Hyperparameters are chosen
on the val cut. The test cut is never used here.

Usage:
    python -m src.train                          # full model
    python -m src.train --no-forgetting          # ablation: NLGCL without forgetting
    python -m src.train --no-contrastive         # ablation: LightGCN + forgetting
    python -m src.train --no-contrastive --no-forgetting   # plain LightGCN
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from src.build_graph import Cut, load_cuts
from src.config import ARTIFACTS, EVENTS_FILE, SEED, TOPICS_FILE
from src.evaluate import evaluate_scores
from src.model import GraphTensors, ReviseMalin


def make_pairs(cut: Cut, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One (student, failed topic, correct topic) triple per failed candidate, negatives resampled each call."""
    users, pos, neg = [], [], []
    for u, (topics, failed) in cut.eligible().items():
        bad, good = topics[failed == 1], topics[failed == 0]
        users.append(np.full(len(bad), u))
        pos.append(bad)
        neg.append(rng.choice(good, size=len(bad)))
    return np.concatenate(users), np.concatenate(pos), np.concatenate(neg)


@torch.no_grad()
def pair_loss(model: ReviseMalin, g: GraphTensors, cut: Cut, rng: np.random.Generator) -> float:
    """BPR loss on a cut's pairs, used as the validation loss curve.

    Only the BPR term: the contrastive term compares every pair with every
    other one, which would not fit in memory on a whole cut.
    """
    u, p, n = (torch.as_tensor(a) for a in make_pairs(cut, rng))
    emb = model.propagate(g)[:2]
    diff = model.score(g, u, p, emb) - model.score(g, u, n, emb)
    return float(-torch.nn.functional.logsigmoid(diff).mean())


def train(args: argparse.Namespace) -> dict:
    """Run training and save the model and its history in artifacts/."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    events = pd.read_parquet(EVENTS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, len(pd.read_parquet(TOPICS_FILE))
    cuts = load_cuts(events, n_users, n_topics)
    g_train, g_val = GraphTensors.from_cut(cuts["train"]), GraphTensors.from_cut(cuts["val"])

    model = ReviseMalin(
        n_users, n_topics, dim=args.dim, n_layers=args.layers,
        use_contrastive=not args.no_contrastive, use_forgetting=not args.no_forgetting,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best, best_state, wait = [], -1.0, None, 0
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("full" if not (args.no_contrastive or args.no_forgetting) else
                       f"{'nocl' if args.no_contrastive else 'cl'}_{'noforget' if args.no_forgetting else 'forget'}")
    model_path, history_path = ARTIFACTS / f"model_{tag}.pt", ARTIFACTS / f"history_{tag}.json"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        u, p, n = make_pairs(cuts["train"], rng)
        order = rng.permutation(len(u))
        model.train()
        losses = []
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            out = model.loss(
                g_train, torch.as_tensor(u[idx]), torch.as_tensor(p[idx]), torch.as_tensor(n[idx]),
                reg_weight=args.reg, cl_reg=args.cl_reg,
            )
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            losses.append(float(out["bpr"].detach()))
        model.eval()
        val = evaluate_scores(model.score_all(g_val).numpy(), cuts["val"])
        row = {
            "epoch": epoch,
            "train_bpr": float(np.mean(losses)),
            "val_bpr": pair_loss(model, g_val, cuts["val"], rng),
            "val_ndcg@10": val["ndcg@10"],
            "beta": float(model.beta.detach()),
            "median_half_life": float(model.half_life().median()),
            "seconds": round(time.time() - start, 2),
        }
        history.append(row)
        print(json.dumps(row))
        # Save the history every epoch and the model at every new best, so a stopped run loses nothing.
        history_path.write_text(json.dumps(history, indent=2))
        if row["val_ndcg@10"] > best:
            best, wait = row["val_ndcg@10"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"state": best_state, "config": vars(args), "epoch": epoch}, model_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}, best val NDCG@10 = {best:.4f}")
                break

    model.load_state_dict(best_state)
    return {"best_val_ndcg@10": best, "epochs": len(history), "tag": tag}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options; defaults follow the NLGCL paper and official config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--cl-reg", type=float, default=1e-3,
                        help="weight of the batch-mean contrastive term (best of 1e-5, 1e-4, 1e-3 on validation)")
    parser.add_argument("--no-contrastive", action="store_true")
    parser.add_argument("--no-forgetting", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(train(parse_args()))
