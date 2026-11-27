"""Final evaluation: trained models against the baselines, with bootstrap intervals.

Every model and baseline ranks the same candidates (topics the student attempted
in the label window) for the same eligible students, with the harness of
src/evaluate.py. A trained model keeps the embeddings it learned on the train
cut and propagates them on the graph built from history strictly before the
evaluated cut, so no answer of the label window reaches its inputs.

The report answers two Week 3 questions:
    Q2 (sanity floor): does each model beat random and most-popular (topics
        answered most often before the cut) on NDCG@10 and Recall@20? "Yes"
        needs the lower bound of the 95% bootstrap interval of the gap to be above 0.
        Two harder references are also reported: personal error rate and recent
        activity (answers per topic in the student's last WINDOW answers).
    Q4 (innovation proof): the full model against each ablation, on the main
        metrics and on Recall@10 for topics not practised for > 14 days.

Usage:
    python -m src.evaluate_test                    # every artifacts/model_*.pt, test cut
    python -m src.evaluate_test --models artifacts/model_full.pt artifacts/model_cl_noforget.pt
    python -m src.evaluate_test --cut val          # same table on the validation cut

Writes reports/<cut>_results.md and reports/<cut>_results.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.build_graph import Cut, build_cut
from src.config import ARTIFACTS, EVENTS_FILE, LABEL, ROOT, SEED, TOPICS_FILE
from src.evaluate import baseline_scores, bootstrap_gap, evaluate_scores, recent_activity_scores
from src.model import GraphTensors, ReviseMalin

METRICS = ["ndcg@10", "recall@20", "recall@5", "recall@10_old"]
GAP_METRICS = ["ndcg@10", "recall@20", "recall@10_old"]
BASELINES = ["random", "most_popular", "global_error", "personal_error", "recent_activity"]
SANITY_REFS = ["random", "most_popular"]  # the Week 3 Q2 floor: random and most-popular (attempt frequency)
SANITY_METRICS = ["ndcg@10", "recall@20"]
LABELS = {
    "random": "Random (mean of seeds)",
    "most_popular": "Most-popular (attempt frequency)",
    "global_error": "Global error rate",
    "personal_error": "Personal error rate",
    "recent_activity": "Recent activity (answers in the last window)",
    "full": "Full model (NLGCL + forgetting)",
    "cl_noforget": "NLGCL, no forgetting",
    "nocl_forget": "LightGCN + forgetting",
    "nocl_noforget": "LightGCN",
}


def load_model(path: Path, n_users: int, n_topics: int) -> tuple[str, ReviseMalin]:
    """Rebuild a model saved by src.train from its checkpoint.

    Args:
        path: A model_<tag>.pt file written by src.train.
        n_users: Number of students in the processed data.
        n_topics: Number of topics in the processed data.

    Returns:
        The tag (file name without "model_") and the model in eval mode.
    """
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = ReviseMalin(
        n_users, n_topics, dim=cfg["dim"], n_layers=cfg["layers"],
        use_contrastive=not cfg["no_contrastive"], use_forgetting=not cfg["no_forgetting"],
        use_topic_bias=cfg.get("topic_bias", False),
    )
    state = dict(ckpt["state"])
    state.setdefault("topic_bias", torch.zeros(n_topics))  # checkpoints trained before the topic bias existed
    model.load_state_dict(state)
    model.eval()
    return path.stem.removeprefix("model_"), model


def _average_runs(runs: list[dict]) -> dict:
    """Average several evaluate_scores results (random seeds), per student included."""
    out = {k: float(np.mean([r[k] for r in runs])) for k in METRICS + ["students", "students_old"]}
    out["per_student"] = {k: np.mean([r["per_student"][k] for r in runs], axis=0) for k in runs[0]["per_student"]}
    return out


def evaluate_cut(cut: Cut, models: dict[str, ReviseMalin], n_random_seeds: int = 10,
                 extra_baselines: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """Evaluate the baselines and every trained model on one cut.

    Args:
        cut: The cut whose label window is evaluated.
        models: Trained models by tag.
        n_random_seeds: Seeds averaged for the random baseline.
        extra_baselines: Score matrices of baselines that need more than the cut (e.g. recent_activity).

    Returns:
        {name: evaluate_scores result}, baselines first.
    """
    results = {"random": _average_runs([evaluate_scores(baseline_scores(cut, "random", s), cut)
                                        for s in range(n_random_seeds)])}
    for name in ["most_popular", "global_error", "personal_error"]:
        results[name] = evaluate_scores(baseline_scores(cut, name), cut)
    for name, scores in (extra_baselines or {}).items():
        results[name] = evaluate_scores(scores, cut)
    g = GraphTensors.from_cut(cut)
    for name, model in models.items():
        results[name] = evaluate_scores(model.score_all(g).numpy(), cut)
    return results


def gap_rows(results: dict[str, dict], model_names: list[str], n_boot: int = 1000, seed: int = SEED) -> list[dict]:
    """Paired bootstrap gaps: each model against each baseline, then the full model against each ablation.

    Args:
        results: Output of evaluate_cut.
        model_names: Tags of the trained models.
        n_boot: Bootstrap resamples over students.
        seed: Seed of the resampling.

    Returns:
        One row per (comparison, metric) with the mean gap and its 95% interval.
    """
    pairs = [(m, b) for m in model_names for b in BASELINES if b in results]
    if "full" in model_names:
        pairs += [("full", m) for m in model_names if m != "full"]
    rows = []
    for a, b in pairs:
        for metric in GAP_METRICS:
            x, y = results[a]["per_student"][metric], results[b]["per_student"][metric]
            if len(x) == 0:
                continue
            assert len(x) == len(y), "both rankings must cover the same students"
            mean, lo, hi = bootstrap_gap(x, y, n=n_boot, seed=seed)
            rows.append({"model": a, "reference": b, "metric": metric, "gap": mean, "ci_low": lo, "ci_high": hi,
                         "significant": bool(lo > 0 or hi < 0)})
    return rows


def sanity_floor(gaps: list[dict], model_names: list[str]) -> dict[str, bool]:
    """Week 3 Q2: the model beats random and most-popular on NDCG@10 and Recall@20 (interval above 0)."""
    out = {}
    for m in model_names:
        rows = [r for r in gaps if r["model"] == m and r["reference"] in SANITY_REFS and r["metric"] in SANITY_METRICS]
        out[m] = len(rows) == len(SANITY_REFS) * len(SANITY_METRICS) and all(r["ci_low"] > 0 for r in rows)
    return out


def forgetting_params(models: dict[str, ReviseMalin]) -> dict[str, dict]:
    """Learned beta, never-seen value and half-lives of the models that use forgetting."""
    out = {}
    for name, model in models.items():
        if model.use_forgetting:
            h = model.half_life().detach().numpy()
            out[name] = {
                "beta": float(model.beta.detach()), "unseen": float(model.unseen.detach()),
                "half_life_days_median": float(np.median(h)),
                "half_life_days_p10": float(np.percentile(h, 10)),
                "half_life_days_p90": float(np.percentile(h, 90)),
            }
    return out


def _fmt(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.4f}"


def render_markdown(cut_name: str, results: dict[str, dict], model_names: list[str], gaps: list[dict],
                    floor: dict[str, bool], params: dict[str, dict], header: str) -> str:
    """Markdown report: metrics table, gaps with intervals, sanity floor and forgetting parameters."""
    lines = [f"# Results on the {cut_name} cut", "", header, ""]
    first = results["random"]
    lines += [f"Eligible students: {int(first['students'])} "
              f"({int(first['students_old'])} with a failed topic not practised for > 14 days).", ""]
    lines += ["| Model | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for name in [b for b in BASELINES if b in results] + model_names:
        lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(_fmt(results[name][k]) for k in METRICS) + " |")

    lines += ["", "## Gaps (paired bootstrap over students, 95% interval)", "",
              "| Model | Reference | Metric | Gap | 95% CI | Significant |", "|---|---|---|---|---|---|"]
    for r in gaps:
        lines.append(f"| {LABELS.get(r['model'], r['model'])} | {LABELS.get(r['reference'], r['reference'])} | "
                     f"{r['metric']} | {r['gap']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{'yes' if r['significant'] else 'no'} |")

    lines += ["", "## Sanity floor (Week 3 Q2)", "",
              "Beats random and most-popular on NDCG@10 and Recall@20, with the 95% interval of every gap above 0.", ""]
    lines += [f"- {LABELS.get(m, m)}: **{'yes' if ok else 'no'}**" for m, ok in floor.items()]

    if params:
        lines += ["", "## Learned forgetting parameters", "",
                  "beta > 0 means a topic gets riskier the longer it has not been practised.", "",
                  "| Model | beta | never-seen value | half-life median (days) | p10 | p90 |", "|---|---|---|---|---|---|"]
        for name, p in params.items():
            lines.append(f"| {LABELS.get(name, name)} | {p['beta']:+.4f} | {p['unseen']:+.4f} | "
                         f"{p['half_life_days_median']:.2f} | {p['half_life_days_p10']:.2f} | "
                         f"{p['half_life_days_p90']:.2f} |")
    return "\n".join(lines) + "\n"


def to_json(results: dict[str, dict], gaps: list[dict], floor: dict[str, bool], params: dict[str, dict],
            meta: dict) -> dict:
    """JSON-friendly version of the report (no per-student arrays)."""
    metrics = {name: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in res.items() if k != "per_student"} for name, res in results.items()}
    return {"meta": meta, "metrics": metrics, "gaps": gaps, "sanity_floor": floor, "forgetting": params}


def main() -> None:
    """Evaluate every trained model and the baselines on one cut, then write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, nargs="*", default=None,
                        help="model_<tag>.pt files (default: every artifacts/model_*.pt)")
    parser.add_argument("--cut", default="test", choices=["val", "test"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--label", choices=["above_mean", "any_fail"], default=LABEL,
                        help="label of the evaluated window (any_fail writes <cut>_results_any_fail.*)")
    args = parser.parse_args()

    paths = args.models if args.models else sorted(ARTIFACTS.glob("model_*.pt"))
    if not paths:
        raise SystemExit("No trained model found in artifacts/: run python -m src.train first.")

    start = time.time()
    events = pd.read_parquet(EVENTS_FILE)
    n_users, n_topics = int(events["user"].max()) + 1, len(pd.read_parquet(TOPICS_FILE))
    cut = build_cut(events, args.cut, n_users, n_topics, rule=args.label)
    order = list(LABELS)
    models = dict(sorted((load_model(p, n_users, n_topics) for p in paths),
                         key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
    names = list(models)

    results = evaluate_cut(cut, models, extra_baselines={"recent_activity": recent_activity_scores(events, cut)})
    gaps = gap_rows(results, names, n_boot=args.bootstrap, seed=args.seed)
    floor = sanity_floor(gaps, names)
    params = forgetting_params(models)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "no git"
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:  # say so, instead of citing a commit that does not hold the code that ran
        commit += " + uncommitted changes in src/"
    meta = {"cut": args.cut, "label": args.label, "commit": commit, "date": time.strftime("%Y-%m-%d %H:%M"),
            "models": {n: str(p) for n, p in zip(names, paths)}, "bootstrap": args.bootstrap, "seed": args.seed,
            "seconds": round(time.time() - start, 1)}
    header = (f"Commit `{commit}`, {meta['date']}. Label: `{args.label}`. Models: {', '.join(names)}. "
              f"{args.bootstrap} bootstrap resamples, seed {args.seed}.")

    md = render_markdown(args.cut, results, names, gaps, floor, params, header)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.cut}_results" + ("" if args.label == LABEL else f"_{args.label}")
    (args.out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(to_json(results, gaps, floor, params, meta), indent=2), encoding="utf-8")
    print(md)
    print(f"Saved to {args.out_dir / f'{stem}.md'} and .json")


if __name__ == "__main__":
    main()
