"""E8 -- is Proposal B's gain structural, or just a smaller step?

E6 found partitioned n=1 gaining +0.094 pass@16 (+2.9 sem) over global iid at no
measurable pass@1 cost. Before that counts, the null explanation has to be killed,
and it is the same one that killed the last two positive results in this project:

    With n_active=1 of P=3 parts, a member's perturbation carries roughly 1/P the
    squared norm of a global one. E1c established that a smaller step slows
    commitment and preserves pass@k, and that plain hyperparameters reproduce the
    entire effect of every mechanism that appeared to work.

So: sweep the global arm's step size across (and past) the partitioned arm's
effective step. If any global setting reproduces the pass@16 gain, partitioning per
se does nothing.

Arms:
    global, fixed sigma in {0.0029, 0.005, 0.0087, 0.015}   -- 0.005/sqrt(3) = 0.0029
                                                               is the norm-matched one
    global, resolution rule                                 -- E6's reference arm
    partitioned n=1, fixed sigma 0.005                      -- matched to the sweep
    partitioned n=1, resolution rule                        -- E6's winning arm

Fixed sigma for the sweep because the resolution rule adapts each arm independently,
so an adaptive comparison cannot isolate step size -- which is also a gap in E6:
sigma was never logged per arm, so the arms may not have been at comparable steps in
either direction. This run logs it.

Run:  python code/experiments/e8_partition_control.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.core.trainer import TrainConfig, Trainer
from evolve.objectives.countdown_obj import CountdownObjective
from evolve.operators import sampling, sigma, weighting

SEEDS = list(range(8))
GENS = 250
BASE_SIGMA = 0.005
P = 3                                   # emb, h, out


def run(kind, seed, sig, adaptive):
    obj = CountdownObjective(n_problems=96, seed=seed)
    smp = (sampling.PartitionedSampler(n_active=1) if kind == "part"
           else sampling.IIDSampler())
    cfg = TrainConfig(n_pop=128, rank=1, sigma=sig, alpha=0.001,
                      generations=GENS, seed=seed)
    rule = sigma.ResolutionRule(sig) if adaptive else None
    tr = Trainer(obj, smp, weighting.RankWeighting(), cfg, sigma_rule=rule)
    tr.run()
    tr.close()
    rep = obj.report(ks=(1, 16), seed=seed)
    final_sigma = tr.history[-1].sigma
    return rep["pass@1"], rep["pass@16"], final_sigma


ARMS = [
    (f"global σ={BASE_SIGMA/P**0.5:.4f} (norm-matched)", "global", BASE_SIGMA / P**0.5, False),
    (f"global σ={BASE_SIGMA}",                           "global", BASE_SIGMA, False),
    (f"global σ={BASE_SIGMA*3:.4f}",                     "global", BASE_SIGMA * 3, False),
    ("global + resolution",                              "global", BASE_SIGMA, True),
    (f"partitioned σ={BASE_SIGMA}",                      "part",   BASE_SIGMA, False),
    ("partitioned + resolution",                         "part",   BASE_SIGMA, True),
]


def main():
    print(f"E8  is Proposal B structural or step size?   seeds={len(SEEDS)} gens={GENS}")
    print(f"norm-matched global sigma = {BASE_SIGMA}/sqrt({P}) = {BASE_SIGMA/P**0.5:.4f}")
    print("E6 reference: global 0.1406, partitioned n=1 0.2344 pass@16\n")

    rows = []
    for label, kind, sig, adaptive in ARMS:
        t0 = time.time()
        res = [run(kind, s, sig, adaptive) for s in SEEDS]
        p1 = np.array([r[0] for r in res])
        pk = np.array([r[1] for r in res])
        sg = np.array([r[2] for r in res])
        rows.append((label, p1.mean(), p1.std(ddof=1) / len(p1) ** .5,
                     pk.mean(), pk.std(ddof=1) / len(pk) ** .5, sg.mean()))
        print(f"  {label:34s} {time.time()-t0:6.1f}s  pass@1 {p1.mean():.4f}  "
              f"pass@16 {pk.mean():.4f}")

    print(f"\n{'arm':34s} {'pass@1':>8s} {'sem':>6s} {'pass@16':>8s} {'sem':>6s} "
          f"{'final σ':>9s}")
    for label, m1, s1, mk, sk, sg in rows:
        print(f"{label:34s} {m1:>8.4f} {s1:>6.3f} {mk:>8.4f} {sk:>6.3f} {sg:>9.5f}")

    best_global = max((r for r in rows if r[0].startswith("global")), key=lambda r: r[3])
    part = [r for r in rows if r[0].startswith("partitioned + res")][0]
    d = part[3] - best_global[3]
    se = (part[4] ** 2 + best_global[4] ** 2) ** 0.5
    print(f"\npartitioned vs the BEST global arm ({best_global[0]}):")
    print(f"  pass@16 {part[3]:.4f} vs {best_global[3]:.4f}   Δ={d:+.4f}  {d/se:+.1f} sem")
    print("\nIf the best global step matches partitioned, the E6 result was step size.")
    print("Only a Δ that survives against the best-tuned global arm is structural.")


if __name__ == "__main__":
    main()
