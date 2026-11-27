"""Metric values checked by hand, and the harness on mock predictions."""
import numpy as np
import pytest

from src.build_graph import build_cut
from src.evaluate import baseline_scores, evaluate_scores, ndcg_at_k, recall_at_k
from tests.conftest import N_TOPICS


def test_ndcg_by_hand():
    assert ndcg_at_k(np.array([1, 0, 0, 0.0]), 10) == pytest.approx(1.0)
    assert ndcg_at_k(np.array([0, 1, 0, 0.0]), 10) == pytest.approx(1 / np.log2(3))
    expected = (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert ndcg_at_k(np.array([0, 1, 1, 0.0]), 10) == pytest.approx(expected)


def test_recall_by_hand():
    assert recall_at_k(np.array([0, 1, 1, 0.0]), 2) == 0.5
    assert recall_at_k(np.array([0, 0.0]), 2) == 0.0


def test_mock_predictions_return_metrics(events):
    cut = build_cut(events, "test", 3, N_TOPICS)
    res = evaluate_scores(np.random.default_rng(0).normal(size=(3, N_TOPICS)), cut)
    assert 0 <= res["ndcg@10"] <= 1 and res["students"] > 0


def test_oracle_scores_perfectly(events):
    cut = build_cut(events, "test", 3, N_TOPICS)
    oracle = np.zeros((3, N_TOPICS))
    oracle[cut.cand_user, cut.cand_topic] = cut.cand_failed
    res = evaluate_scores(oracle, cut)
    assert res["ndcg@10"] == pytest.approx(1.0)


def test_recent_activity_only_counts_answers_before_the_cut(events):
    from src.build_graph import cut_times
    from src.evaluate import recent_activity_scores

    cut = build_cut(events, "test", 3, N_TOPICS)
    counts = np.floor(recent_activity_scores(events, cut, window=50))
    start = events["user"].map(cut_times(events)["test"])
    past = events[events["timestamp"] < start]
    for u in range(3):
        last = past[past["user"] == u].nsmallest(50, "answer_idx")
        assert counts[u].sum() == len(last)  # exactly the 50 answers right before the cut, none after


def test_baselines_have_the_right_shape(events):
    cut = build_cut(events, "test", 3, N_TOPICS)
    for name in ["random", "most_popular", "global_error", "personal_error"]:
        assert baseline_scores(cut, name).shape == (3, N_TOPICS)
