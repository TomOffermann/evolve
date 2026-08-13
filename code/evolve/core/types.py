"""Core protocols. The seams where EA components plug in.

[P1](research/problems/P1-continuous-diversity.md) identified three injection points
for evolutionary machinery, and the E0-E4 experiments confirmed they behave very
differently. They are separate interfaces here so that a component cannot quietly
occupy two of them at once:

    Sampler    -- how perturbations are drawn        (sampling)
    Weighting  -- how member fitness becomes update weight  (fitness shaping / selection)
    SigmaRule  -- how the step size adapts           (step size)

`Objective` owns the model and the data, because only it knows how to run a forward
pass. The trainer owns the ES logic and never sees a dense parameter delta.

Two API decisions are evidence-driven and worth stating:

1. `Evaluation.per_example` is **required**, not optional. The Tier-0 signature --
   double-centred per-example fitness -- is the only diversity metric that survived
   E0 (rho = 0.76-0.87 vs a floor of 0.00), and it is free because the objective
   computes it anyway. Making it mandatory means every diagnostic and every
   diversity operator works on every objective, with no per-objective plumbing.

2. `EvalRequest.crn_seed` carries **common random numbers**. In E2, stochastic
   objectives evaluated with independent per-member sampling had their ES gradient
   estimated almost entirely from sampling noise, and training ran backwards. Any
   stochastic objective must derive its randomness from this seed so that member
   differences reflect weight differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

Tensor = torch.Tensor
Factors = dict[str, tuple[Tensor, Tensor]]   # name -> (A: (N,m,r), B: (N,n,r))
Shapes = dict[str, tuple[int, int]]


@dataclass
class EvalRequest:
    """One chunk of a population, handed to the objective for evaluation."""

    factors: Factors
    sigma: float
    rank: int
    crn_seed: int          # shared across the WHOLE generation -- see module docstring
    generation: int
    chunk_id: int = 0
    n_chunks: int = 1

    @property
    def size(self) -> int:
        A, _ = next(iter(self.factors.values()))
        return A.shape[0]


@dataclass
class Evaluation:
    """What the objective returns for a chunk."""

    fitness: Tensor                      # (N,)   scalar fitness per member
    per_example: Tensor                  # (N, M) the Tier-0 signature. Required.
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.per_example is None:
            raise ValueError(
                "per_example is required: it is the Tier-0 signature every "
                "diversity operator and diagnostic is built on, and the objective "
                "computes it anyway. Return the per-example scores you already have."
            )
        if self.per_example.shape[0] != self.fitness.shape[0]:
            raise ValueError(
                f"per_example has {self.per_example.shape[0]} rows but fitness has "
                f"{self.fitness.shape[0]}; they must agree on population size."
            )


@runtime_checkable
class Objective(Protocol):
    """Owns the model parameters and the data."""

    @property
    def shapes(self) -> Shapes:
        """Perturbable matrices: name -> (out, in)."""

    def evaluate(self, req: EvalRequest) -> Evaluation: ...

    def apply_update(self, delta: dict[str, Tensor]) -> None:
        """Add `delta[name]` to each perturbable matrix in place."""

    def parent_fitness(self, req: EvalRequest) -> float:
        """Unperturbed fitness under the SAME crn_seed. Used by sigma rules.

        Must be paired with the population evaluation -- an unpaired parent makes
        the success/tie statistic pure noise (E2 Finding 3)."""


@runtime_checkable
class Sampler(Protocol):
    """Injection point 1: how perturbations are drawn."""

    def draw(self, seed: int, shapes: Shapes, n_pop: int, rank: int,
             device) -> Factors: ...

    def importance(self, factors: Factors, n_pop: int) -> Tensor | None:
        """Per-member 1/pi_i correction, or None if the sampler is unbiased.

        Non-uniform samplers (e.g. partitioned masks, P2) MUST return this or the
        ES gradient estimator is biased. The bias is silent and it poisons any
        bandit built on top of the utilities."""


@runtime_checkable
class Weighting(Protocol):
    """Injection point 2: fitness -> per-member update weight."""

    def weights(self, fitness: Tensor, per_example: Tensor,
                state: dict) -> Tensor: ...


@runtime_checkable
class SigmaRule(Protocol):
    """Injection point 3: step-size adaptation."""

    sigma: float

    def update(self, fitness: Tensor, parent_fitness: float) -> float: ...


@runtime_checkable
class Diagnostic(Protocol):
    """Read-only observer. Must never influence the update."""

    def observe(self, gen: int, fitness: Tensor, per_example: Tensor,
                sigma: float, objective: Objective) -> dict: ...
