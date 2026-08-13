"""Injection point 1: how perturbations are drawn.

The cheapest place to win, and the only one where diversity can *reduce* estimator
variance rather than trading against fitness. It is also the injection point that
[Proposal B](research/proposals/B-modular-partition-diversity.md) lives at -- the
one substantive idea in the programme that is still untested.

Every sampler must implement `importance()`. Uniform samplers return None; any
sampler that draws members non-uniformly MUST return the per-member 1/pi_i
correction or the ES gradient estimator is silently biased.
"""

from __future__ import annotations

import torch

from ..core.noise import draw_factors
from ..core.types import Factors, Shapes


class IIDSampler:
    """Plain iid Gaussian factors. EGGROLL's own choice and the default."""

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0) -> Factors:
        return draw_factors(seed, shapes, n_pop, rank, device,
                            index_offset=index_offset)

    def importance(self, factors, n_pop):
        return None


class AntitheticSampler:
    """Mirrored pairs: E and -E. Since -A B^T = A (-B)^T, mirroring is a sign flip.

    **Measured worse than iid** (E1: -0.18), provisionally -- that run carried an
    RNG-ordering flaw and deserves a clean rerun. The mechanism is plausible though:
    at a fixed evaluation budget, mirroring halves the number of *independent*
    exploration directions (N -> N/2), and in high dimension the direction count
    dominates the control-variate benefit. That may be why EGGROLL samples
    independently despite the variate being free."""

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0) -> Factors:
        half = draw_factors(seed, shapes, (n_pop + 1) // 2, rank, device,
                            index_offset=index_offset)
        out = {}
        for k, (A, B) in half.items():
            out[k] = (torch.cat([A, A])[:n_pop], torch.cat([B, -B])[:n_pop])
        return out

    def importance(self, factors, n_pop):
        return None


class OrthogonalSampler:
    """Orthogonalise the A factors within each chunk (Choromanski et al.).

    Forcing exploration directions to be exactly orthogonal yields an estimator with
    **strictly lower MSE** than iid -- a theorem, not a heuristic. This is the one
    surviving claim from Proposal C after E0 falsified its diversity story, and it
    has never been tested here. Judge it on progress-per-rollout, not on any
    diversity criterion.

    Only meaningful while n_pop <= out_dim; beyond that the rows cannot all be
    mutually orthogonal and this degrades gracefully to iid."""

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0) -> Factors:
        fac = draw_factors(seed, shapes, n_pop, rank, device,
                           index_offset=index_offset)
        out = {}
        for name, (A, B) in fac.items():
            N, m, r = A.shape
            if N <= m:
                flat = A.reshape(N, m * r)
                q, _ = torch.linalg.qr(flat.T)                 # (m*r, N)
                scale = flat.norm(dim=1, keepdim=True)
                A = (q.T * scale).reshape(N, m, r)
            out[name] = (A, B)
        return out

    def importance(self, factors, n_pop):
        return None


class PartitionedSampler:
    """Perturb only a subset of the network per member. **Proposal B / P2.**

    Each member draws a mask over `parts` and is perturbed only there. This gives
    the genotype a *discrete* component again -- a bitstring of length P -- on which
    Hamming distance applies unmodified, which is the metric DEGA actually uses and
    the one that does not survive the move to R^d.

    Three things it buys beyond diversity:
      * credit assignment -- a P-dimensional signal instead of d-dimensional noise
      * per-part utilities you can run a bandit over (the "higher-level parameters")
      * approximate additivity across disjoint parts, so one rollout carries
        information about several hypotheses

    **The importance correction is not optional.** With part p sampled at
    probability pi_p, member weights must be scaled by 1/pi_p or the estimator is
    biased -- silently, and in a way that poisons any bandit built on the utilities.

    **Validated (decision 0006).** The first mechanism in this project to clear a
    properly tuned control:

      E8 (8 seeds): pass@16 0.2319 vs 0.1567 for the best global arm across a 5x
      sigma sweep -- +5.0 sem. Step size does not explain it.
      E9 (5 seeds): at matched pass@1 the advantage is positive at every checkpoint
      and grows with training (+0.047 at pass@1 0.110). Global's pass@16 falls
      monotonically 0.233 -> 0.139; partitioned's stays flat at ~0.22.

    **Not free.** Costs ~1.8 sem of pass@1 and reaches a given pass@1 later than
    global. Use it when pass@k matters; do not use it to optimise pass@1 alone.
    The frontiers coincide at low pass@1 -- the advantage appears in the upper range.

    Measured at 94k parameters, one task, 3 parts. Unknown at scale."""

    def __init__(self, parts: dict[str, list[str]] | None = None,
                 n_active: int = 1, probs: dict[str, float] | None = None,
                 floor: float = 0.05):
        self.parts = parts          # part name -> list of matrix names; None = per matrix
        self.n_active = n_active
        self.probs = probs
        self.floor = floor
        self._last_mask: torch.Tensor | None = None
        self._part_names: list[str] = []

    def _resolve(self, shapes: Shapes) -> dict[str, list[str]]:
        return self.parts if self.parts else {k: [k] for k in shapes}

    def _pi(self, names) -> torch.Tensor:
        if self.probs is None:
            p = torch.full((len(names),), 1.0 / len(names))
        else:
            p = torch.tensor([self.probs.get(n, self.floor) for n in names])
            p = p.clamp_min(self.floor)
            p = p / p.sum()
        return p

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0) -> Factors:
        parts = self._resolve(shapes)
        names = list(parts)
        self._part_names = names
        pi = self._pi(names)

        # Mask is keyed on GLOBAL member index too, so chunking cannot change it.
        mask = torch.zeros(n_pop, len(names))
        for j in range(n_pop):
            g = torch.Generator(device="cpu").manual_seed(
                (int(seed) ^ 0x5EED) + 0x9E37 * (index_offset + j))
            pick = torch.multinomial(pi, self.n_active, replacement=False, generator=g)
            mask[j, pick] = 1.0
        self._last_mask = mask.to(device)
        self._last_pi = pi.to(device)

        fac = draw_factors(seed, shapes, n_pop, rank, device,
                           index_offset=index_offset)
        for j, part in enumerate(names):
            keep = mask[:, j].to(device).view(-1, 1, 1)
            for mat in parts[part]:
                A, B = fac[mat]
                fac[mat] = (A * keep, B)          # zeroing A zeroes E = A B^T
        return fac

    def importance(self, factors, n_pop):
        """1/pi for the part each member actually perturbed."""
        if self._last_mask is None:
            return None
        # per-member effective probability = sum of pi over its active parts
        pi_eff = (self._last_mask * self._last_pi).sum(dim=1).clamp_min(1e-8)
        return 1.0 / (pi_eff * len(self._part_names))

    def mask_hamming(self) -> torch.Tensor | None:
        """Pairwise normalised Hamming distance between member masks.

        This is DEGA's metric, unmodified, on a genotype that actually has one."""
        if self._last_mask is None:
            return None
        m = self._last_mask
        P = m.shape[1]
        return (m.unsqueeze(1) != m.unsqueeze(0)).float().sum(-1) / P
