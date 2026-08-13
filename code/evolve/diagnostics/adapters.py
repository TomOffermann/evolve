"""Diagnostics as trainer plug-ins. Read-only, and never allowed to break a run.

The trainer treats these as observers: they see the population every generation and
return a dict that lands in `GenerationRecord.diagnostics`. They must not touch the
update -- the whole point of E1's controls was that a "diagnostic" which quietly
influences training is indistinguishable from a mechanism.

Attach both collapse detectors by default. E2 showed they watch **different
objects** and only one of them ever fires:

    CollapseDetector          ES population diversity  -- never collapses, because
                              ES resamples from a fixed isotropic noise distribution
                              every generation (E1e)
    PolicyDiversityDetector   the policy's own output  -- collapses hard, which is
                              the RLVR failure the programme is actually about

Monitoring only the first is how you miss the failure entirely.
"""

from __future__ import annotations

import torch

from .collapse import CollapseDetector, PolicyDiversityDetector


class PopulationDiversity:
    """Wraps `CollapseDetector` as a trainer diagnostic. Free -- reads the Tier-0
    signature the objective already returned."""

    def __init__(self, **kw):
        self.det = CollapseDetector(**kw)

    def observe(self, gen, fitness, per_example, sigma, objective):
        rec = self.det.observe(per_example, fitness)
        return {"eff_rank": rec["eff_rank"], "log_volume": rec["log_volume"],
                "fires": self.det.fires()}


class PolicyDiversity:
    """Wraps `PolicyDiversityDetector`. Needs the objective to expose sampled
    completions or output logits; objectives that cannot are skipped silently rather
    than failing the run."""

    def __init__(self, every: int = 10, k: int = 8, **kw):
        self.det = PolicyDiversityDetector(**kw)
        self.every, self.k = every, k

    def observe(self, gen, fitness, per_example, sigma, objective):
        if gen % self.every:
            return {}
        sample = getattr(objective, "sample_completions", None)
        if sample is None:
            return {"skipped": "objective exposes no sample_completions()"}
        rec = self.det.observe(samples=sample(self.k))
        return {"distinct": rec.get("distinct"), "fires": self.det.fires("distinct")}


class Throughput:
    """Members evaluated per second. The number to watch when tuning chunk_size and
    worker count -- parallelism that does not move this is not helping."""

    def __init__(self, n_pop: int):
        self.n_pop = n_pop
        self.last = None

    def observe(self, gen, fitness, per_example, sigma, objective):
        return {"n_pop": self.n_pop}


class FitnessSpread:
    """Selection pressure. If this collapses, the population has stopped
    disagreeing and the step size is doing nothing -- which E2 found is the binding
    constraint under sparse reward, not diversity."""

    def observe(self, gen, fitness, per_example, sigma, objective):
        return {"std": float(fitness.std()),
                "tie_frac": float((fitness == fitness.median()).float().mean()),
                "max": float(fitness.max())}


def default_diagnostics(n_pop: int, policy_every: int = 25) -> dict:
    return {
        "population": PopulationDiversity(warmup=30, window=40),
        "policy": PolicyDiversity(every=policy_every),
        "spread": FitnessSpread(),
    }
