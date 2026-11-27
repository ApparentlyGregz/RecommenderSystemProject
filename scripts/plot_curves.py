"""Plot the training and validation curves of every run for the Week 3 report.

Reads artifacts/history_<tag>.json (written every epoch by src.train) or, failing
that, the JSON lines of artifacts/train_<tag>.log. The full model trained with the
old summed InfoNCE is drawn in dashed gray: it shows the bug the batch mean fixed.

Usage:
    python scripts/plot_curves.py              ->  reports/figures/training_curves_week3.png (EdNet)
    python scripts/plot_curves.py --artifacts artifacts/dbe --out reports/dbe/figures/training_curves.png
    python scripts/plot_curves.py --compact    ->  ..._compact.png, sized for a 7-inch report column
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "reports" / "figures" / "training_curves_week3.png"

# Fixed categorical order (validated palette), one color per model whatever runs exist.
RUNS = [
    ("full", "Full (NLGCL + forgetting)", "#2a78d6", "-"),
    ("cl_noforget", "NLGCL, no forgetting", "#eb6834", "-"),
    ("nocl_forget", "LightGCN + forgetting", "#1baf7a", "-"),
    ("nocl_noforget", "LightGCN", "#eda100", "-"),
    ("full_sumloss", "Full, old summed InfoNCE (bug)", "#888780", "--"),
]
PANELS = [("train_bpr", "BPR loss, train"), ("val_bpr", "BPR loss, validation"), ("val_ndcg@10", "NDCG@10, validation")]


def load(tag: str) -> list[dict] | None:
    """History of one run, or None if it does not exist."""
    hist = ARTIFACTS / f"history_{tag}.json"
    if hist.exists():
        return json.loads(hist.read_text())
    log = ARTIFACTS / f"train_{tag}.log"
    if log.exists():
        rows = [json.loads(l) for l in log.read_text(errors="ignore").splitlines() if l.startswith('{"epoch"')]
        return rows or None
    return None


def main() -> None:
    global ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compact", action="store_true", help="figure sized for a 7-inch report column")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS, help="folder with history_<tag>.json")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    ARTIFACTS = args.artifacts
    out = args.out.with_name(args.out.stem + "_compact.png") if args.compact else args.out
    size, font, title_font, legend_font, lw, ms = ((7.2, 2.45), 7, 7.5, 6, 1.4, 4.5) if args.compact else \
        ((15, 4.2), 10, 11, 9, 2, 7)

    runs = [(tag, label, color, ls, h) for tag, label, color, ls in RUNS if (h := load(tag))]
    if not runs:
        sys.exit("No training history found in artifacts/.")
    plt.rcParams.update({"font.size": font, "axes.edgecolor": "#b4b2a9", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 3, figsize=size, facecolor="#fcfcfb")
    for ax, (key, title) in zip(axes, PANELS):
        ax.set_facecolor("#fcfcfb")
        for _, label, color, ls, h in runs:
            x = [r["epoch"] for r in h]
            y = [r[key] for r in h]
            ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=label)
            if key == "val_ndcg@10":
                best = max(range(len(y)), key=y.__getitem__)
                ax.plot(x[best], y[best], "o", markersize=ms, color=color, markeredgecolor="#fcfcfb",
                        markeredgewidth=lw)
        ax.set_title(title, loc="left", color="#0b0b0b", fontsize=title_font)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[2].legend(frameon=False, fontsize=legend_font, loc="lower right")
    if not args.compact:
        fig.text(0.01, 0.01, "Dots: best validation epoch (checkpoint kept). Same split, seed 42, "
                 "early stopping on validation only.", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1) if not args.compact else (0, 0, 1, 1), pad=0.4 if args.compact else 1.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300 if args.compact else 160, facecolor=fig.get_facecolor())
    for tag, label, _, _, h in runs:
        best = max(h, key=lambda r: r["val_ndcg@10"])
        print(f"{label:34s} epochs {len(h):3d} | best val NDCG@10 {best['val_ndcg@10']:.4f} at epoch {best['epoch']}"
              f" | final train BPR {h[-1]['train_bpr']:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
