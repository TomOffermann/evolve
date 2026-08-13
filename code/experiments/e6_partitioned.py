"""E6 -- Proposal B: does partitioned perturbation do anything?

The one substantive idea in the programme that has never been tested. Every
mechanism E0-E4 killed was a population-*reweighting* scheme; this one changes
**what gets perturbed**, so none of those negative results apply to it.

Each member perturbs only a subset of the network. That gives the genotype a
discrete component -- a mask over parts -- on which Hamming distance applies
unmodified, and it gives per-part credit assignment, which global ES cannot provide
at all.

Arms:
    global (iid)          the reference
    partitioned n=1       one part per member; maximum structural separation
    partitioned n=2       two parts per member
    interleaved           global and partitioned generations alternating -- Tom's
                          own suggestion, and DEGA's phase structure

Reported alongside fitness: **per-part utility**, which is the interpretability
artefact partitioned ES gives away free. Even if no arm wins, that trace answers a
question global ES cannot pose -- which parts of the network are currently learnable.

Run:  python code/experiments/e6_partitioned.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.core.trainer import TrainConfig, Trainer
from evolve.objectives.countdown_obj import CountdownObjective
from evolve.operators import sampling, sigma, weighting

SEEDS = [0, 1, 2]
GENS = 250


class PartUtility:
    """Diagnostic: mean |centred fitness| of the members that perturbed each part.

    This is the P-dimensional credit-assignment signal from P2. Global ES gives a
    d-dimensional one, which is noise; this one you can plot."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.totals: dict[int, float] = {}
        self.counts: dict[int, int] = {}

    def observe(self, gen, fitness, per_example, sig, objective):
        mask = getattr(self.sampler, "_last_mask", None)
        if mask is None:
            return {}
        f = (fitness - fitness.mean()).abs()
        for p in range(mask.shape[1]):
            sel = mask[:, p] > 0
            if sel.any():
                self.totals[p] = self.totals.get(p, 0.0) + float(f[sel].sum())
                self.counts[p] = self.counts.get(p, 0) + int(sel.sum())
        return {}

    def utilities(self, names):
        return {names[p]: self.totals[p] / max(self.counts[p], 1)
                for p in sorted(self.totals)}


class InterleavedSampler:
    """Alternate global and partitioned generations.

    DEGA's phase structure, in the one place it might actually port: exploit with
    full-network steps, explore structure with partitioned ones. E4 showed DEGA's
    *weighting* transplants fail; this tests whether its *scheduling* idea fares
    better at a different injection point."""

    def __init__(self, period=2, n_active=1):
        self.glob = sampling.IIDSampler()
        self.part = sampling.PartitionedSampler(n_active=n_active)
        self.period = period
        self._active = self.glob
        self._gen = 0

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0):
        self._active = self.part if (self._gen % self.period) else self.glob
        return self._active.draw(seed, shapes, n_pop, rank, device,
                                 index_offset=index_offset)

    def importance(self, factors, n_pop):
        return self._active.importance(factors, n_pop)

    def tick(self):
        self._gen += 1


def run(arm, seed):
    obj = CountdownObjective(n_problems=96, seed=seed)
    if arm == "global (iid)":
        smp = sampling.IIDSampler()
    elif arm.startswith("partitioned"):
        smp = sampling.PartitionedSampler(n_active=int(arm[-1]))
    else:
        smp = InterleavedSampler(period=2, n_active=1)

    util = PartUtility(smp.part if isinstance(smp, InterleavedSampler) else smp)
    cfg = TrainConfig(n_pop=128, rank=1, sigma=0.005, alpha=0.001,
                      generations=GENS, seed=seed)
    tr = Trainer(obj, smp, weighting.RankWeighting(), cfg,
                 sigma_rule=sigma.ResolutionRule(0.005),
                 diagnostics={"util": util})
    if isinstance(smp, InterleavedSampler):
        for g in range(GENS):
            tr.step(g)
            smp.tick()
    else:
        tr.run()
    tr.close()
    rep = obj.report(ks=(1, 16), seed=seed)
    return rep["pass@1"], rep["pass@16"], util, list(obj.shapes)


ARMS = ["global (iid)", "partitioned 1", "partitioned 2", "interleaved"]


def main():
    print(f"E6  Proposal B: partitioned perturbation   N=128 gens={GENS} "
          f"seeds={len(SEEDS)}")
    obj0 = CountdownObjective(seed=0)
    base = obj0.report(ks=(1, 16), seed=0)
    print(f"parts = {list(obj0.shapes)}")
    print(f"SFT base: pass@1 {base['pass@1']:.4f}  pass@16 {base['pass@16']:.4f}\n")

    rows, utils = [], {}
    for arm in ARMS:
        t0 = time.time()
        res = [run(arm, s) for s in SEEDS]
        p1 = np.array([r[0] for r in res])
        pk = np.array([r[1] for r in res])
        rows.append((arm, p1.mean(), p1.std(ddof=1) / len(p1) ** .5,
                     pk.mean(), pk.std(ddof=1) / len(pk) ** .5))
        utils[arm] = res[0][2].utilities(res[0][3])
        print(f"  {arm:16s} {time.time()-t0:5.1f}s  pass@1 {p1.mean():.4f}  "
              f"pass@16 {pk.mean():.4f}")

    b = rows[0]
    print(f"\n{'arm':16s} {'pass@1':>8s} {'sem':>6s} {'Δ':>8s} {'pass@16':>8s} "
          f"{'sem':>6s} {'Δ':>8s}")
    for arm, m1, s1, mk, sk in rows:
        print(f"{arm:16s} {m1:>8.4f} {s1:>6.3f} {m1-b[1]:>+8.4f} "
              f"{mk:>8.4f} {sk:>6.3f} {mk-b[3]:>+8.4f}")

    print("\nper-part utility (seed 0) -- the signal global ES cannot produce:")
    for arm, u in utils.items():
        if u:
            tot = sum(u.values()) or 1.0
            share = "  ".join(f"{k}={v/tot:.2f}" for k, v in u.items())
            print(f"  {arm:16s} {share}")

    print("\nΔ vs global. A partitioned arm is interesting if it holds pass@1 while")
    print("producing a usable per-part utility trace; matching is already a result,")
    print("since the utilities come free and global ES has no equivalent.")


if __name__ == "__main__":
    main()
