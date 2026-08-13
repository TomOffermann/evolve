"""Injection point 2: fitness -> per-member update weight.

Every operator here has been benchmarked. The verdict is in its docstring, with the
regime it applies to, because E0-E4 showed the answer flips between regimes: on
dense reward with no collapse, nothing beats plain ranks; on sparse verifiable
reward where the policy genuinely collapses, the Tier-0 novelty bonus wins.

Anything that composes with `RankWeighting` composes by mixing *ranks*, not raw
fitness -- so the mixing weight lambda means the same thing regardless of how the
objective scales its reward.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


# --------------------------------------------------------------------- primitives

def centred_ranks(F: Tensor) -> Tensor:
    """Centred rank transform with **tied ranks averaged**.

    The naive version (argsort -> distinct ranks) is fine for dense fitness and
    actively harmful under sparse binary reward, where many members score exactly
    the same: it breaks those ties by argsort order, which is arbitrary, and injects
    noise into the update in proportion to how sparse the reward is. E2 found this
    the hard way."""
    N = F.numel()
    order = F.argsort()
    ranks = torch.empty_like(F)
    idx = torch.arange(N, dtype=F.dtype, device=F.device)
    _, inv, counts = torch.unique(F[order], return_inverse=True, return_counts=True)
    sums = torch.zeros(counts.numel(), dtype=F.dtype, device=F.device)
    sums.index_add_(0, inv, idx)
    ranks[order] = (sums / counts)[inv]
    return ranks / max(N - 1, 1) - 0.5


def double_centre(f: Tensor) -> Tensor:
    """Remove per-example difficulty AND per-member skill.

    Without both, the signature measures 'who is good' rather than 'who is
    different' -- which is the one thing a diversity metric must not do (E0)."""
    return f - f.mean(0, keepdim=True) - f.mean(1, keepdim=True) + f.mean()


def signature(f: Tensor) -> Tensor:
    """Tier 0: double-centred, row-normalised per-example fitness.

    Validated in E0 at rho = 0.76-0.87 against held-out behavioural decorrelation,
    versus a floor of 0.00 for every parameter-space metric tested."""
    z = double_centre(f)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------- operators

class RankWeighting:
    """Plain centred ranks. The baseline, and the best choice on dense objectives
    with no measured collapse (E1c: step size dominates every mechanism there)."""

    def weights(self, fitness, per_example, state):
        return centred_ranks(fitness)


class NoveltyBonus:
    """k-NN novelty in Tier-0 signature space, mixed into the fitness ranking.

    **NOT ESTABLISHED as better than noise.** E4 reported +2.7 sem over a random
    control, but that control had a fixed RNG seed and so drew the identical bonus
    sequence in every run -- one realization replicated 8 times, with error bars that
    excluded its own variability. Rerun against a properly randomised control
    (E7, 8 seeds): the margin falls to **+1.1 sem at lambda=0.2 and +1.7 sem at
    lambda=0.4**, both under the 2-sem bar, and at lambda=0.4 it *trades* pass@1 for
    pass@16 rather than dominating.

    What IS established: **some** bonus raises pass@16 over no mechanism at all
    (0.148 -> 0.168 at lambda=0.2, -> 0.201 at lambda=0.4), and novelty is
    directionally ahead of noise at both lambdas tested. That consistency is weak
    positive evidence, not a result.

    **Not recommended on dense objectives with no collapse** -- E1b found a random
    bonus matched it there, because the only thing any bonus can do in that regime is
    mimic a smaller step size.

    lambda > 0.5 is harmful in both regimes (E1b: lambda=0.7 cost 1.3 nats)."""

    def __init__(self, lam: float = 0.3, k: int = 10):
        self.lam, self.k = lam, k

    def weights(self, fitness, per_example, state):
        S = signature(per_example)
        d = 1.0 - S @ S.T
        d.fill_diagonal_(float("inf"))
        k = min(self.k, max(d.shape[0] - 1, 1))
        nov = d.topk(k, largest=False).values.mean(dim=1)
        return (1 - self.lam) * centred_ranks(fitness) + self.lam * centred_ranks(nov)


class RandomBonus:
    """CONTROL for NoveltyBonus: same mixing weight, pure noise instead of signal.

    Keep this in the library and run it in every diversity comparison. Without it,
    'novelty helps' is unfalsifiable -- any mechanism that partly decouples the
    update from fitness injects exploration, and that alone can look like a win.
    E1b used exactly this arm to overturn E1's headline result.

    **Its noise must vary with the run seed.** The first version used a fixed
    constant, so every seed of a multi-seed comparison drew the *identical* bonus
    sequence -- measuring one realization replicated N times, and reporting a
    between-seed sem that did not include the control's own variability at all.
    E4's headline "+2.7 sem" was computed against such a control and is not
    trustworthy as stated. The trainer seeds `state["run_seed"]`; use it.
    """

    def __init__(self, lam: float = 0.3, seed: int = 1234):
        self.lam, self.seed = lam, seed

    def weights(self, fitness, per_example, state):
        g = state.setdefault("_rand_bonus_rng", torch.Generator().manual_seed(
            self.seed ^ (int(state.get("run_seed", 0)) * 2654435761 & 0x7FFFFFFF)))
        r = torch.rand(fitness.shape, generator=g).to(fitness.device)
        return (1 - self.lam) * centred_ranks(fitness) + self.lam * centred_ranks(r)


class AdaptiveNovelty:
    """NSRA-ES: raise lambda while stagnating, lower it on progress.

    **Measured worse than fixed lambda** (E1: -0.58 vs fixed novelty). The schedule
    ratchets up during stagnation and recovers too slowly, so it spends most of
    training optimising novelty over fitness. Kept because the failure is
    instructive: adaptive diversity weights are where these methods usually die."""

    def __init__(self, k: int = 10, lo: float = 0.0, hi: float = 0.8, step: float = 0.05):
        self.k, self.lo, self.hi, self.step = k, lo, hi, step

    def weights(self, fitness, per_example, state):
        lam = state.get("_ns_lam", 0.2)
        best, prev = float(fitness.max()), state.get("_ns_best", -1e30)
        if best > prev + 1e-4:
            lam, state["_ns_best"] = max(self.lo, lam - self.step), best
        else:
            lam = min(self.hi, lam + self.step)
        state["_ns_lam"] = lam
        return NoveltyBonus(lam, self.k).weights(fitness, per_example, state)


class NicheBalanced:
    """One vote per behavioural niche instead of one vote per member.

    **Measured neutral** (E1d: +0.004 +/- 0.006 on top of tuned hyperparameters).
    The a-priori argument was the best in the programme -- rare-behaviour members are
    outvoted -- and it turned out to be true but irrelevant, because the population
    is behaviourally diverse anyway (E1e). You cannot fix a representation problem
    that does not exist.

    `hard=True` equalises total weight per niche and also destroys the between-niche
    ranking; it was much worse early in training. Prefer the soft inverse-propensity
    form, and anneal it in rather than applying it from step zero."""

    def __init__(self, n_niches: int = 8, hard: bool = False, lloyd_steps: int = 5):
        self.n_niches, self.hard, self.lloyd_steps = n_niches, hard, lloyd_steps

    def weights(self, fitness, per_example, state):
        S = signature(per_example)
        N = S.shape[0]
        k = min(self.n_niches, N)
        idx = [int(torch.argmax(S.norm(dim=1)))]
        for _ in range(k - 1):
            idx.append(int(torch.argmax(1.0 - (S @ S[idx].T).max(dim=1).values)))
        C = S[idx].clone()
        assign = (S @ C.T).argmax(dim=1)
        for _ in range(self.lloyd_steps):
            assign = (S @ C.T).argmax(dim=1)
            for c in range(k):
                m = assign == c
                if m.any():
                    C[c] = S[m].mean(0)
            C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)

        w = centred_ranks(fitness)
        if self.hard:
            out = torch.zeros_like(w)
            for c in range(k):
                m = assign == c
                if m.any():
                    wc = w[m] - w[m].mean()
                    n = wc.norm()
                    if n > 1e-9:
                        out[m] = wc / n
        else:
            counts = torch.bincount(assign, minlength=k).clamp_min(1).to(w.dtype)
            out = w / counts[assign].sqrt()
        return out / out.norm().clamp_min(1e-12) * (N**0.5 * 0.29)


class RedundancyWhitened:
    """GLS on correlated observations: w <- (K + ridge I)^-1 w.

    **Measured worse** (E1: -0.30). The idea was that members with correlated
    signatures double-count the same evidence, so whitening should reduce variance.
    In practice inverting a noisy NxN correlation matrix amplifies exactly the
    directions the signature estimates least reliably. Diversity-as-variance-
    reduction does not work here; diversity-as-exploration does."""

    def __init__(self, ridge: float = 0.1):
        self.ridge = ridge

    def weights(self, fitness, per_example, state):
        S = signature(per_example)
        N = S.shape[0]
        K = S @ S.T + self.ridge * torch.eye(N, device=S.device, dtype=S.dtype)
        w = torch.linalg.solve(K, centred_ranks(fitness))
        return w / w.norm().clamp_min(1e-12) * (N**0.5 * 0.29)


class DipecPartial:
    """DEGA's subsampled improving step: keep each member with prob 1/lambda.

    **Does not port to ES.** Swept properly in E4: monotone degradation, no sweet
    spot (pass@1 0.122 -> 0.079 as lambda goes 1 -> 8), with lambda -> 1 recovering
    the baseline as it must.

    The reason is structural. In DEGA the improving mask has meaningful components,
    so dropping some keeps a valid smaller improvement. In ES the population *is* a
    Monte-Carlo gradient estimate, so subsampling multiplies its variance by roughly
    lambda and buys nothing."""

    def __init__(self, lam: float = 2.0, seed: int = 0):
        self.lam, self.seed = lam, seed

    def weights(self, fitness, per_example, state):
        g = state.setdefault("_dipec_rng", torch.Generator().manual_seed(
            self.seed ^ (int(state.get("run_seed", 0)) * 2654435761 & 0x7FFFFFFF)))
        w = centred_ranks(fitness)
        keep = (torch.rand(w.shape, generator=g).to(w.device) < 1.0 / self.lam)
        return w * keep.to(w.dtype) * self.lam


class Compose:
    """Average several weightings. Useful for ablations; no measured winner yet."""

    def __init__(self, *parts, weights=None):
        self.parts = parts
        self.mix = weights or [1.0 / len(parts)] * len(parts)

    def weights(self, fitness, per_example, state):
        out = None
        for p, m in zip(self.parts, self.mix):
            w = p.weights(fitness, per_example, state) * m
            out = w if out is None else out + w
        return out
