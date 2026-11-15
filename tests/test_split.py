"""The split must never let a label window feed its own inputs."""
import numpy as np

from src.build_graph import build_cut, cut_times
from src.config import CUTS
from tests.conftest import N_TOPICS


def test_inputs_are_strictly_before_each_cut(events):
    times = cut_times(events)
    for name in CUTS:
        cut = build_cut(events, name, 3, N_TOPICS)
        start = events["user"].map(times[name])
        assert cut.attempts.sum() == (events["timestamp"] < start).sum()
        assert np.nanmin(cut.practice_dt) > 0
        assert cut.edge_dt.min() > 0


def test_test_window_never_used_as_input(events):
    times = cut_times(events)
    test_start = events["user"].map(times["test"])
    n_test_events = (events["timestamp"] >= test_start).sum()
    for name in CUTS:
        cut = build_cut(events, name, 3, N_TOPICS)
        assert cut.attempts.sum() <= len(events) - n_test_events


def test_bundles_are_not_split(events):
    times = cut_times(events)
    for name in CUTS:
        start = events["user"].map(times[name])
        before = events[events["timestamp"] < start]
        after = events[events["timestamp"] >= start]
        shared = set(zip(before["user"], before["timestamp"])) & set(zip(after["user"], after["timestamp"]))
        assert not shared


def test_labels_are_attempted_topics(events):
    cut = build_cut(events, "test", 3, N_TOPICS)
    assert set(np.unique(cut.cand_failed)) <= {0, 1}
    assert len(cut.cand_user) > 0
