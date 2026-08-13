"""E5 -- does the framework reproduce the hand-rolled E2/E4 results?

A rewrite that changes the numbers is not a refactor, it is a new experiment with
unknown provenance. Before the framework is used for anything, it has to land on the
E2/E4 reference points.

Reference (countdown post-training, 250 gens, resolution rule):

    SFT base                pass@1 ~0.055   pass@16 ~0.211
    resolution rule         pass@1  0.1237  pass@16 0.1641   (3 seeds, E2)
    resolution + novelty .2 pass@1  0.1068  pass@16 0.2018   (3 seeds, E2)
    novelty .2 vs RANDOM .2 pass@16 0.1880 vs 0.1479         (8 seeds, E4)

Exact equality is not expected -- the framework draws noise keyed on global member
index where the old code keyed it on a per-generation seed, so the populations
differ member for member. What must reproduce is the *ordering and the size of the
effects*: novelty holds pass@k, the random control does not, and neither costs much
pass@1.

Also checks that chunked execution changes nothing, which is the property that makes
large-N runs trustworthy.

Run:  python code/experiments/e5_framework_regression.py
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


def run_arm(weight_op, seed, chunk_size=None, gens=None):
    obj = CountdownObjective(n_problems=96, seed=seed)
    cfg = TrainConfig(n_pop=128, rank=1, sigma=0.005, alpha=0.001,
                      generations=gens or GENS, chunk_size=chunk_size, seed=seed)
    tr = Trainer(obj, sampling.IIDSampler(), weight_op, cfg,
                 sigma_rule=sigma.ResolutionRule(0.005))
    tr.run()
    tr.close()
    rep = obj.report(ks=(1, 16), seed=seed)
    return rep["pass@1"], rep["pass@16"], tr


ARMS = [
    ("resolution (no mech)", lambda: weighting.RankWeighting()),
    ("+ novelty λ=0.2",      lambda: weighting.NoveltyBonus(lam=0.2)),
    ("+ RANDOM λ=0.2 (ctrl)", lambda: weighting.RandomBonus(lam=0.2)),
]


def main():
    obj0 = CountdownObjective(seed=0)
    base = obj0.report(ks=(1, 16), seed=0)
    print(f"E5  framework regression   params={obj0.n_params()}  "
          f"N=128 gens={GENS} seeds={len(SEEDS)}")
    print(f"SFT base: pass@1 {base['pass@1']:.4f}  pass@16 {base['pass@16']:.4f}\n")

    rows = []
    for label, make in ARMS:
        t0 = time.time()
        res = [run_arm(make(), s) for s in SEEDS]
        p1 = np.array([r[0] for r in res])
        pk = np.array([r[1] for r in res])
        rows.append((label, p1.mean(), p1.std(ddof=1) / len(p1) ** .5,
                     pk.mean(), pk.std(ddof=1) / len(pk) ** .5))
        print(f"  {label:22s} {time.time()-t0:5.1f}s  pass@1 {p1.mean():.4f}  "
              f"pass@16 {pk.mean():.4f}")

    print(f"\n{'arm':22s} {'pass@1':>8s} {'sem':>6s} {'pass@16':>8s} {'sem':>6s} {'gap':>7s}")
    for label, m1, s1, mk, sk in rows:
        print(f"{label:22s} {m1:>8.4f} {s1:>6.3f} {mk:>8.4f} {sk:>6.3f} {mk-m1:>7.4f}")

    nov = [r for r in rows if "novelty" in r[0]][0]
    ctl = [r for r in rows if "RANDOM" in r[0]][0]
    none = rows[0]
    print("\nreference (E2/E4, hand-rolled code):")
    print("  resolution        pass@1 0.1237  pass@16 0.1641")
    print("  + novelty λ=0.2   pass@1 0.1068  pass@16 0.2018")
    print("  novelty vs RANDOM pass@16 0.1880 vs 0.1479 (8 seeds)")
    print("\nchecks:")
    print(f"  collapse reproduced (pass@16 falls below base): "
          f"{none[3] < base['pass@16']}")
    print(f"  novelty holds pass@16 above no-mechanism:       {nov[3] > none[3]}")
    print(f"  novelty beats its RANDOM control on pass@16:    {nov[3] > ctl[3]}")

    print("\n--- chunking invariance (seed 0, 40 gens) ---")
    a = run_arm(weighting.RankWeighting(), 0, chunk_size=None, gens=40)[:2]
    b = run_arm(weighting.RankWeighting(), 0, chunk_size=32, gens=40)[:2]
    print(f"  unchunked   pass@1 {a[0]:.4f}  pass@16 {a[1]:.4f}")
    print(f"  chunk=32    pass@1 {b[0]:.4f}  pass@16 {b[1]:.4f}")
    print(f"  identical:  {a == b}")


if __name__ == "__main__":
    main()
