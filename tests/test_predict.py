"""Serving predictor: same scores as the PyTorch model, and the time term behaves as designed."""
import json

import numpy as np
import torch

from src.model import GraphTensors, ReviseMalin
from src.predict import Predictor

N_USERS, N_TOPICS = 10, 20


def fake_serving(path, seed: int = 0):
    """Write a small serving.npz from a randomly initialised model (no training needed)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ReviseMalin(N_USERS, N_TOPICS, dim=8)
    with torch.no_grad():
        model.beta.fill_(0.8)
        model.unseen.fill_(0.1)
        model.log_half_life.copy_(torch.log(torch.linspace(2, 30, N_TOPICS)))
    practice_dt = rng.uniform(0, 40, (N_USERS, N_TOPICS)).astype(np.float32)
    practice_dt[rng.random((N_USERS, N_TOPICS)) < 0.2] = np.nan
    eu, et = np.nonzero(rng.random((N_USERS, N_TOPICS)) < 0.3)
    g = GraphTensors(N_USERS, N_TOPICS, torch.as_tensor(eu), torch.as_tensor(et),
                     torch.as_tensor(rng.uniform(0, 30, len(eu)).astype(np.float32)), torch.as_tensor(practice_dt))
    with torch.no_grad():
        u, t, _ = model.propagate(g)
        ref = model.score_all(g).numpy()
    attempts = np.where(np.isnan(practice_dt), 0, rng.integers(1, 50, (N_USERS, N_TOPICS)))
    np.savez_compressed(
        path, user_emb=u.numpy(), topic_emb=t.numpy(), half_life=model.half_life().detach().numpy(),
        beta=np.float32(0.8), unseen=np.float32(0.1), use_forgetting=np.bool_(True), practice_dt=practice_dt,
        attempts=attempts.astype(np.uint16), fails=(attempts // 3).astype(np.uint16),
        user_id=np.arange(N_USERS) + 1000, n_answers=np.full(N_USERS, 500, np.int32),
        level_attempts=np.full((N_USERS, 3), 30, np.uint16), level_fails=np.full((N_USERS, 3), 9, np.uint16),
        topic_kc=np.arange(N_TOPICS) + 1, topic_name=np.array([f"notion {i}" for i in range(N_TOPICS)]),
        topic_description=np.array([f"about notion {i}" for i in range(N_TOPICS)]),
        topic_related=np.array(["2;3"] + [""] * (N_TOPICS - 1)),
        meta=np.array(json.dumps({"model": "test"})),
    )
    np.save(str(path) + ".ref.npy", ref)
    return path


def test_numpy_scores_match_torch(tmp_path):
    path = fake_serving(tmp_path / "s.npz")
    pred = Predictor.load(path)
    ref = np.load(str(path) + ".ref.npy")
    assert np.allclose(pred.score_matrix(), ref, atol=1e-5)


def test_waiting_raises_risk_of_old_topics_when_beta_positive(tmp_path):
    pred = Predictor.load(fake_serving(tmp_path / "s.npz"))
    now = {r["topic"]: r["time_score"] for r in pred.recommend(0, k=N_TOPICS)}
    later = {r["topic"]: r["time_score"] for r in pred.recommend(0, k=N_TOPICS, days_ahead=30)}
    seen = [t for t in now if not np.isnan(pred.practice_dt[0, t])]
    assert all(later[t] > now[t] for t in seen)  # beta > 0: risk grows with time since practice
    assert any("forgotten" in r["reason"] for r in pred.recommend(0, k=N_TOPICS))


def test_negative_beta_is_explained_without_forgetting(tmp_path):
    pred = Predictor.load(fake_serving(tmp_path / "s.npz"))
    pred.beta = -0.5
    assert not any("forgotten" in r["reason"] for r in pred.recommend(0, k=N_TOPICS))


def test_labels_and_related(tmp_path):
    pred = Predictor.load(fake_serving(tmp_path / "s.npz"))
    assert pred.topic_label(0) == "Notion 0"
    assert pred.related(0) == ["Notion 1", "Notion 2"]  # KC ids 2 and 3 are topics 1 and 2


def test_practised_only_and_hidden(tmp_path):
    pred = Predictor.load(fake_serving(tmp_path / "s.npz"))
    recs = pred.recommend(1, k=N_TOPICS, practised_only=True, hidden={0, 1})
    assert all(pred.attempts[1, r["topic"]] > 0 for r in recs)
    assert not {0, 1} & {r["topic"] for r in recs}
