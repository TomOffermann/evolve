"""ES Objective wrapping the MoE router for EGGROLL-style optimization.

The objective exposes only the router weights as perturbable matrices. Expert
weights, embedding, and output head are frozen. ES perturbs the router to find
routing decisions that maximise task accuracy -- a discrete decision (argmax)
that ES handles without the softmax relaxation gradient methods need.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure evolve and local modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolve.core.types import EvalRequest, Evaluation, Shapes

from model import ToyMoE, make_modular_arithmetic_data


class MoERouterObjective:
    """Objective that perturbs only the router weight matrix.

    Fitness = fraction of validation examples classified correctly under hard
    (argmax) routing. Per-example scores are binary correct/incorrect per sample.
    """

    def __init__(self, model: ToyMoE, x_val: torch.Tensor, y_val: torch.Tensor,
                 n_eval: int = 512, device: str = "cpu", seed: int = 0):
        self.model = model.to(device)
        self.model.freeze_experts()
        self.device = device
        self.seed = seed
        self.n_eval = n_eval

        self.x_val = x_val[:n_eval].to(device)
        self.y_val = y_val[:n_eval].to(device)

        # Cache the router weight for perturbation
        self._router_w = self.model.router.weight.data  # (K, H)

    @property
    def shapes(self) -> Shapes:
        """Single perturbable matrix: the router weight."""
        K, H = self._router_w.shape
        return {"router": (K, H)}

    def _apply_perturbation(self, A: torch.Tensor, B: torch.Tensor,
                            sigma: float, rank: int) -> torch.Tensor:
        """Compute perturbed router weight: W + sigma * (1/sqrt(r)) A B^T."""
        s = rank ** -0.5
        delta = torch.einsum("mr,nr->mn", A, B) * s * sigma  # (K, H)
        return self._router_w + delta

    def evaluate(self, req: EvalRequest) -> Evaluation:
        N = req.size
        A, B = req.factors["router"]   # A: (N, K, r), B: (N, H, r)

        # Subsample validation set with CRN for consistency across members
        g = torch.Generator().manual_seed(req.crn_seed)
        M = min(self.n_eval, self.x_val.shape[0])
        idx = torch.randperm(self.x_val.shape[0], generator=g)[:M]
        x = self.x_val[idx]
        y = self.y_val[idx]

        per_example = torch.zeros(N, M, device=self.device)

        with torch.no_grad():
            # Precompute shared representations (frozen layers)
            one_hot = F.one_hot(x.long(), self.model.vocab_size).float()
            h = self.model.embedding(one_hot).mean(dim=1)  # (M, H)

            for i in range(N):
                # Perturbed router
                w_pert = self._apply_perturbation(A[i], B[i], req.sigma, req.rank)

                # Hard routing with perturbed weights
                routing_logits = h @ w_pert.T                   # (M, K)
                chosen = routing_logits.argmax(dim=-1)          # (M,)

                out = torch.zeros_like(h)
                for k in range(self.model.n_experts):
                    mask = chosen == k
                    if mask.any():
                        out[mask] = self.model.experts[k](h[mask])

                logits = self.model.output_head(out + h)
                preds = logits.argmax(dim=-1)
                per_example[i] = (preds == y).float()

        fitness = per_example.mean(dim=1)
        return Evaluation(fitness=fitness, per_example=per_example)

    def parent_fitness(self, req: EvalRequest) -> float:
        """Unperturbed fitness under the same CRN seed."""
        g = torch.Generator().manual_seed(req.crn_seed)
        M = min(self.n_eval, self.x_val.shape[0])
        idx = torch.randperm(self.x_val.shape[0], generator=g)[:M]
        x = self.x_val[idx]
        y = self.y_val[idx]

        with torch.no_grad():
            logits = self.model(x, hard=True)
            preds = logits.argmax(dim=-1)
            return float((preds == y).float().mean())

    def apply_update(self, delta: dict[str, torch.Tensor]) -> None:
        self._router_w += delta["router"].to(self._router_w.device)

    # ----------------------------------------------------------------- metrics

    def compute_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """Accuracy, expert utilization, load balance entropy."""
        with torch.no_grad():
            logits, routing_logits = self.model(x, hard=True, return_routing=True)
            preds = logits.argmax(dim=-1)
            accuracy = float((preds == y).float().mean())

            chosen = routing_logits.argmax(dim=-1)
            K = self.model.n_experts
            counts = torch.bincount(chosen, minlength=K).float()
            utilization = counts / counts.sum()

            # Load balance: entropy of routing distribution (max = log(K))
            p = utilization.clamp_min(1e-8)
            entropy = float(-(p * p.log()).sum())
            max_entropy = math.log(K)
            balance = entropy / max_entropy  # 1.0 = perfectly balanced

            # Expert collapse: any expert getting < 1% of traffic
            collapsed = int((utilization < 0.01).sum())

        return {
            "accuracy": accuracy,
            "utilization": utilization.tolist(),
            "entropy": entropy,
            "balance": balance,
            "collapsed_experts": collapsed,
        }


