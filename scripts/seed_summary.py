"""Mean and spread over training seeds for each model variant (validation and test).

On a small dataset two variants can differ by less than two seeds of the same
variant; this table shows whether a gap is larger than that noise.

Usage:
    python scripts/seed_summary.py      ->  reports/dbe/seed_summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.build_graph import build_cut  # noqa: E402
from src.config import ARTIFACTS, DATASET, EVENTS_FILE, TOPICS_FILE  # noqa: E402
from src.evaluate import baseline_scores, evaluate_scores  # noqa: E402
from src.evaluate_test import LABELS, load_model  # noqa: E402
from src.model import GraphTensors  # noqa: E402

VARIANTS = ["full", "cl_noforget", "nocl_forget", "nocl_noforget"]
SEEDS = {"": 42, "_s1": 1, "_s2": 2}
OUT = ROOT / "reports" / ("dbe" if DATASET == "dbe" else "") / "seed_summary.md"


def main() -> None:
    events = pd.read_parquet(EVENTS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, len(pd.read_parquet(TOPICS_FILE))
    test = build_cut(events, "test", n_users, n_topics)
    g = GraphTensors.from_cut(test)
    rows = []
    for v in VARIANTS:
        val, nd, r5 = [], [], []
        for sfx in SEEDS:
            path = ARTIFACTS / f"model_{v}{sfx}.pt"
            if not path.exists():
                continue
            hist = json.loads((ARTIFACTS / f"history_{v}{sfx}.json").read_text())
            val.append(max(h["val_ndcg@10"] for h in hist))
            res = evaluate_scores(load_model(path, n_users, n_topics)[1].score_all(g).numpy(), test)
            nd.append(res["ndcg@10"])
            r5.append(res["recall@5"])
        if nd:
            rows.append((LABELS.get(v, v), len(nd), np.mean(val), np.std(val), np.mean(nd), np.std(nd),
                         np.mean(r5), np.std(r5)))
    base = {b: evaluate_scores(baseline_scores(test, b), test) for b in ["most_popular", "global_error", "personal_error"]}
    lines = ["# Model variants over training seeds", "",
             f"Seeds {sorted(SEEDS.values())}; mean ± standard deviation. Validation: best epoch (used for early "
             "stopping). Test: the kept checkpoint on the test cut.", "",
             "| Variant | Seeds | Val NDCG@10 | Test NDCG@10 | Test Recall@5 |", "|---|---|---|---|---|"]
    for name, n, vm, vs, tm, ts, rm, rs in rows:
        lines.append(f"| {name} | {n} | {vm:.4f} ± {vs:.4f} | {tm:.4f} ± {ts:.4f} | {rm:.4f} ± {rs:.4f} |")
    lines += ["", "Test baselines (deterministic): " + ", ".join(
        f"{LABELS.get(b, b)} NDCG@10 {r['ndcg@10']:.4f}, Recall@5 {r['recall@5']:.4f}" for b, r in base.items()), ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
