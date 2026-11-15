"""Forward pass, gradients and ablation flags of the model."""
import pytest

torch = pytest.importorskip("torch")

from src.build_graph import build_cut  # noqa: E402
from src.model import GraphTensors, ReviseMalin  # noqa: E402
from tests.conftest import N_TOPICS  # noqa: E402


@pytest.fixture
def graph(events):
    return GraphTensors.from_cut(build_cut(events, "train", 3, N_TOPICS))


def test_forward_pass(graph):
    model = ReviseMalin(3, N_TOPICS)
    users, topics, layers = model.propagate(graph)
    assert users.shape == (3, 64) and topics.shape == (N_TOPICS, 64) and len(layers) == 3
    scores = model.score_all(graph)
    assert scores.shape == (3, N_TOPICS) and torch.isfinite(scores).all()


def test_loss_and_gradients(graph):
    model = ReviseMalin(3, N_TOPICS)
    u, p, n = torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    out = model.loss(graph, u, p, n)
    out["total"].backward()
    assert torch.isfinite(out["total"])
    assert model.beta.grad is not None and model.log_half_life.grad.abs().sum() > 0


def test_ablation_switches_off_forgetting(graph):
    model = ReviseMalin(3, N_TOPICS, use_forgetting=False, use_contrastive=False)
    u, t = torch.tensor([0, 1]), torch.tensor([0, 1])
    assert torch.all(model.time_term(graph, u, t) == 0)
    out = model.loss(graph, u, t, torch.tensor([2, 3]))
    assert float(out["contrastive"]) == 0.0
