"""Révise Malin model: LightGCN + NLGCL contrastive loss + learned forgetting.

NLGCL (Xu et al., RecSys 2025) treats each node and its neighbours at the next
GNN layer as two natural views and adds an InfoNCE loss between them, with no
data augmentation. This file follows the official implementation
(github.com/Jinfeng-Xu/NLGCL) on a student-topic graph, written in plain PyTorch.

Our addition is a forgetting component, in the half-life form 2^(-dt / h) of
Duolingo's Half-Life Regression (Settles & Meeder, ACL 2016):
    1. Propagation: each edge (student failed topic) is weighted by 2^(-dt_error / h_topic).
    2. Score: beta * (1 - 2^(-dt_practice / h_topic)), plus a learned value for never-seen topics.
h (one per topic), beta and the never-seen value are learned with the embeddings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.build_graph import Cut


@dataclass
class GraphTensors:
    """Torch view of a Cut: graph edges and time gaps before the cut."""

    n_users: int
    n_topics: int
    edge_user: torch.Tensor
    edge_topic: torch.Tensor
    edge_dt: torch.Tensor
    practice_dt: torch.Tensor

    @classmethod
    def from_cut(cls, cut: Cut) -> "GraphTensors":
        """Convert a Cut (numpy) to tensors."""
        return cls(
            n_users=cut.n_users,
            n_topics=cut.n_topics,
            edge_user=torch.as_tensor(cut.edge_user, dtype=torch.long),
            edge_topic=torch.as_tensor(cut.edge_topic, dtype=torch.long),
            edge_dt=torch.as_tensor(cut.edge_dt, dtype=torch.float32),
            practice_dt=torch.as_tensor(cut.practice_dt, dtype=torch.float32),
        )


class ReviseMalin(nn.Module):
    """Graph recommender of topics to revise, with learned forgetting.

    Args:
        n_users: Number of students.
        n_topics: Number of topics.
        dim: Embedding size (64 in the NLGCL paper).
        n_layers: Number of propagation layers (2 in the official NLGCL config).
        use_contrastive: Ablation flag for the NLGCL loss.
        use_forgetting: Ablation flag for the forgetting component.
        cl_temp: InfoNCE temperature (0.1 in the official config).
        cl_alpha: Weight between the student-side and topic-side contrastive terms (0.6).
        init_half_life: Starting half-life in days.
        half_life_bounds: Min and max half-life in days.
    """

    def __init__(
        self,
        n_users: int,
        n_topics: int,
        dim: int = 64,
        n_layers: int = 2,
        use_contrastive: bool = True,
        use_forgetting: bool = True,
        cl_temp: float = 0.1,
        cl_alpha: float = 0.6,
        init_half_life: float = 7.0,
        half_life_bounds: tuple[float, float] = (0.5, 365.0),
    ) -> None:
        super().__init__()
        self.n_users, self.n_topics, self.n_layers = n_users, n_topics, n_layers
        self.use_contrastive, self.use_forgetting = use_contrastive, use_forgetting
        self.cl_temp, self.cl_alpha = cl_temp, cl_alpha
        self.half_life_bounds = half_life_bounds
        self.user_emb = nn.Embedding(n_users, dim)
        self.topic_emb = nn.Embedding(n_topics, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.topic_emb.weight)
        self.log_half_life = nn.Parameter(torch.full((n_topics,), math.log(init_half_life)))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.unseen = nn.Parameter(torch.tensor(0.0))

    def half_life(self) -> torch.Tensor:
        """Half-life per topic in days, kept positive and bounded."""
        lo, hi = self.half_life_bounds
        return self.log_half_life.exp().clamp(lo, hi)

    def adjacency(self, g: GraphTensors) -> torch.Tensor:
        """Symmetric normalised adjacency D^-1/2 A D^-1/2 with forgetting-weighted edges."""
        if self.use_forgetting:
            w = torch.pow(2.0, -g.edge_dt / self.half_life()[g.edge_topic])
        else:
            w = torch.ones_like(g.edge_dt)
        n = g.n_users + g.n_topics
        row = torch.cat([g.edge_user, g.edge_topic + g.n_users])
        col = torch.cat([g.edge_topic + g.n_users, g.edge_user])
        val = torch.cat([w, w])
        deg = torch.zeros(n).index_add(0, row, val)
        norm = val / torch.sqrt(deg[row] * deg[col]).clamp_min(1e-12)
        return torch.sparse_coo_tensor(torch.stack([row, col]), norm, (n, n)).coalesce()

    def propagate(self, g: GraphTensors) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """LightGCN propagation: mean of the layer embeddings.

        Returns:
            Final student embeddings, final topic embeddings, and the list of layer outputs.
        """
        adj = self.adjacency(g)
        h = torch.cat([self.user_emb.weight, self.topic_emb.weight])
        layers = [h]
        for _ in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
            layers.append(h)
        out = torch.stack(layers).mean(0)
        return out[: g.n_users], out[g.n_users :], layers

    def time_term(self, g: GraphTensors, users: torch.Tensor, topics: torch.Tensor) -> torch.Tensor:
        """Forgetting part of the score; zero when forgetting is switched off."""
        if not self.use_forgetting:
            return torch.zeros(users.shape[0])
        dt = g.practice_dt[users, topics]
        seen = ~torch.isnan(dt)
        decay = 1 - torch.pow(2.0, -torch.nan_to_num(dt) / self.half_life()[topics])
        return torch.where(seen, self.beta * decay, self.unseen.expand_as(decay))

    def score(
        self,
        g: GraphTensors,
        users: torch.Tensor,
        topics: torch.Tensor,
        emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Risk score of (student, topic) pairs: higher = more likely to fail."""
        u_emb, t_emb = emb if emb is not None else self.propagate(g)[:2]
        dot = (u_emb[users] * t_emb[topics]).sum(-1)
        return dot + self.time_term(g, users, topics)

    @torch.no_grad()
    def score_all(self, g: GraphTensors) -> torch.Tensor:
        """Scores for every (student, topic), shape (n_users, n_topics)."""
        u_emb, t_emb, _ = self.propagate(g)
        users = torch.arange(g.n_users).repeat_interleave(g.n_topics)
        topics = torch.arange(g.n_topics).repeat(g.n_users)
        dot = u_emb @ t_emb.T
        return dot + self.time_term(g, users, topics).view(g.n_users, g.n_topics)

    def _info_nce(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """InfoNCE with in-batch negatives, averaged over the batch.

        The official NLGCL code sums over the batch. On our graph that made the
        contrastive gradient about 700x the BPR gradient at initialisation (normalised
        vectors give large gradients when embeddings are small), so BPR stayed at
        ln 2 and the ranking never trained. The mean keeps both terms on the same
        scale whatever the batch size; cl_reg is tuned on validation for this form.
        """
        v1, v2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        logits = v1 @ v2.T / self.cl_temp
        return -(logits.diagonal() - torch.logsumexp(logits, dim=1)).mean()

    def contrastive_loss(self, layers: list[torch.Tensor], users: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """NLGCL neighbour-layer loss: a node vs its neighbours at the next layer."""
        ego_u, ego_t = layers[0][: self.n_users], layers[0][self.n_users :]
        cl_u = cl_t = torch.tensor(0.0)
        for layer in layers[1:]:
            cur_u, cur_t = layer[: self.n_users], layer[self.n_users :]
            cl_u = cl_u + self._info_nce(cur_t[pos], ego_u[users])
            cl_t = cl_t + self._info_nce(cur_u[users], ego_t[pos])
            ego_u, ego_t = cur_u, cur_t
        return self.cl_alpha * cl_u + (1 - self.cl_alpha) * cl_t

    def loss(
        self,
        g: GraphTensors,
        users: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        reg_weight: float = 1e-4,
        cl_reg: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        """BPR + cl_reg * NLGCL contrastive loss + reg_weight * L2.

        Args:
            g: Inputs before the cut.
            users: Student of each training pair.
            pos: Topic attempted and failed in the label window.
            neg: Topic attempted and always correct in the same window.
            reg_weight: L2 weight (1e-4 in the paper).
            cl_reg: Weight of the batch-mean contrastive term, chosen on validation
                (the official 5e-5 applies to a batch sum, i.e. about 0.2 on a mean of 4096).

        Returns:
            Dict with total, bpr, contrastive and l2 terms.
        """
        u_emb, t_emb, layers = self.propagate(g)
        s_pos = self.score(g, users, pos, (u_emb, t_emb))
        s_neg = self.score(g, users, neg, (u_emb, t_emb))
        bpr = -F.logsigmoid(s_pos - s_neg).mean()
        l2 = (
            self.user_emb(users).pow(2).sum() + self.topic_emb(pos).pow(2).sum() + self.topic_emb(neg).pow(2).sum()
        ) / (2 * users.shape[0])
        cl = self.contrastive_loss(layers, users, pos) if self.use_contrastive else torch.tensor(0.0)
        total = bpr + cl_reg * cl + reg_weight * l2
        return {"total": total, "bpr": bpr, "contrastive": cl, "l2": l2}
