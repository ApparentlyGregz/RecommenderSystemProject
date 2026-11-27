"""Final evaluation script: checkpoints reload, every ranking covers the same students, report renders."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.build_graph import build_cut  # noqa: E402
from src.evaluate_test import (  # noqa: E402
    evaluate_cut,
    forgetting_params,
    gap_rows,
    load_model,
    render_markdown,
    sanity_floor,
)
from src.model import ReviseMalin  # noqa: E402
from tests.conftest import N_TOPICS  # noqa: E402


def save(tmp_path, tag, no_contrastive, no_forgetting):
    torch.manual_seed(0)
    model = ReviseMalin(3, N_TOPICS, use_contrastive=not no_contrastive, use_forgetting=not no_forgetting)
    path = tmp_path / f"model_{tag}.pt"
    config = {"dim": 64, "layers": 2, "no_contrastive": no_contrastive, "no_forgetting": no_forgetting}
    torch.save({"state": model.state_dict(), "config": config}, path)
    return path, model


def test_checkpoint_reloads_the_same_scores(events, tmp_path):
    path, model = save(tmp_path, "full", False, False)
    name, loaded = load_model(path, 3, N_TOPICS)
    from src.model import GraphTensors

    g = GraphTensors.from_cut(build_cut(events, "test", 3, N_TOPICS))
    assert name == "full" and not loaded.training
    assert torch.allclose(model.score_all(g), loaded.score_all(g))


def test_report_end_to_end(events, tmp_path):
    cut = build_cut(events, "test", 3, N_TOPICS)
    models = dict(load_model(save(tmp_path, tag, *flags)[0], 3, N_TOPICS)
                  for tag, flags in [("full", (False, False)), ("cl_noforget", (False, True))])
    results = evaluate_cut(cut, models, n_random_seeds=3)

    n = results["random"]["students"]
    for res in results.values():
        assert len(res["per_student"]["ndcg@10"]) == n and 0 <= res["ndcg@10"] <= 1

    gaps = gap_rows(results, list(models), n_boot=50)
    assert {(r["model"], r["reference"]) for r in gaps} >= {("full", "personal_error"), ("full", "cl_noforget")}
    assert all(r["ci_low"] <= r["gap"] <= r["ci_high"] for r in gaps)

    floor = sanity_floor(gaps, list(models))
    params = forgetting_params(models)
    assert set(floor) == {"full", "cl_noforget"} and set(params) == {"full"}

    md = render_markdown("test", results, list(models), gaps, floor, params, "header")
    assert "Full model (NLGCL + forgetting)" in md and "Sanity floor" in md


def test_gap_with_itself_is_zero(events, tmp_path):
    cut = build_cut(events, "test", 3, N_TOPICS)
    results = evaluate_cut(cut, {}, n_random_seeds=2)
    results["copy"] = results["personal_error"]
    rows = [r for r in gap_rows(results, ["copy"], n_boot=20) if r["reference"] == "personal_error"]
    assert rows and all(np.isclose(r["gap"], 0) and not r["significant"] for r in rows)
