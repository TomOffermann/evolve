"""JEPA on pendulum as an `Objective`. Dense reward, representation collapse.

Complements countdown: sparse reward with *policy* collapse there, dense reward with
*representation* collapse here. Between them the framework is exercised on both
failure modes.

The `rank_bonus` term is the point of using ES at all -- an SVD followed by an
entropy, which you cannot backprop through, and which E3 measured as beating every
differentiable surrogate at its own target (6.72 of 8 vs 2.48-5.24).
"""

from __future__ import annotations

import torch

from ..core.types import EvalRequest, Evaluation
from ..tasks.pendulum_jepa import JEPA, Pendulum, evaluate as jepa_eval, fitness


def rank_bonus(z: torch.Tensor) -> torch.Tensor:
    """Effective rank of the latent batch. Non-differentiable by construction."""
    zc = z - z.mean(dim=1, keepdim=True)
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum(-1))


class JepaObjective:
    def __init__(self, batch=256, latent=8, hidden=64, device="cpu", seed=0,
                 var_weight=1.0, cov_weight=0.5, rank_weight=0.3,
                 mask_action=True):
        self.env = Pendulum()
        self.model = JEPA(latent=latent, hidden=hidden, device=device, seed=seed)
        self.var_weight, self.cov_weight = var_weight, cov_weight
        self.rank_weight, self.mask_action = rank_weight, mask_action

        O, A, S = self.env.trajectories(batch, 2, seed=seed)
        self.obs_t = torch.from_numpy(O[:, 0]).to(device)
        self.obs_t1 = torch.from_numpy(O[:, 1]).to(device)
        self.act = torch.from_numpy(A[:, 0]).to(device)

        Oe, _, Se = self.env.trajectories(512, 1, seed=12345)
        self.ev_obs = torch.from_numpy(Oe[:, 0]).to(device)
        self.ev_state = torch.from_numpy(Se[:, 0]).to(device)

    @property
    def shapes(self):
        return self.model.shapes

    def n_params(self):
        return self.model.n_params()

    def _score(self, n, factors, sigma, rank):
        ot = self.obs_t.unsqueeze(0).expand(n, -1, -1)
        ot1 = self.obs_t1.unsqueeze(0).expand(n, -1, -1)
        ac = self.act.unsqueeze(0).expand(n, -1, -1)
        F, per_ex, z = fitness(self.model, ot, ot1, ac, factors, sigma, rank,
                               var_weight=self.var_weight,
                               cov_weight=self.cov_weight,
                               mask_action=self.mask_action)
        if self.rank_weight:
            bonus = self.rank_weight * rank_bonus(z)
            F = F + bonus
            per_ex = per_ex + bonus[:, None]
        return F, per_ex

    def evaluate(self, req: EvalRequest) -> Evaluation:
        F, per_ex = self._score(req.size, req.factors, req.sigma, req.rank)
        return Evaluation(fitness=F, per_example=per_ex)

    def parent_fitness(self, req: EvalRequest) -> float:
        F, _ = self._score(1, None, 0.0, req.rank)
        return float(F[0])

    def apply_update(self, delta):
        for name, d in delta.items():
            self.model.W[name] += d

    def report(self):
        er, r2, std = jepa_eval(self.model, self.ev_obs, self.ev_state)
        return {"eff_rank": er, "probe_r2": r2, "latent_std": std}
