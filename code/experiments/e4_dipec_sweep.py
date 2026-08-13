"""E4 -- DiPEC subsample rate sweep, on the realistic benchmark.

E1 tested DiPEC at a single lambda=4 and it was catastrophic (-1.80). Reporting a
mechanism as dead from one arbitrary hyperparameter is not a fair test: lambda -> 1
keeps every member and must recover the baseline exactly, so there is a continuum
and the question is where it turns.

DiPEC keeps each contributing member with probability 1/lambda and rescales by
lambda. In DEGA that subsamples a *meaningful* improving mask. In ES the population
is a Monte-Carlo gradient estimate, so the same operator multiplies estimator
variance by roughly lambda. The prediction is therefore a monotone degradation with
no sweet spot -- but that is a prediction, and it is cheap to check.

Also swept: novelty lambda, on the sparse-reward benchmark where E1's dense-reward
conclusions may not carry.

Run:  python code/experiments/e4_dipec_sweep.py
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e2_countdown_posttrain as E2
from evolve.diversity import mechanisms as MECH

E2.GENERATIONS = 250
SEEDS_DIPEC = [0, 1, 2]
SEEDS_NOV = list(range(8))   # the decisive comparison needs tighter error bars


def sweep(name, arms, SEEDS):
    print(f"\n### {name}")
    print(f"{'arm':22s} {'pass@1':>8s} {'sem':>6s} {'pass@16':>8s} {'sem':>6s} {'gap':>7s}")
    for label, fn in arms:
        res = [E2.run(label, fn, s, adapt_sigma="resolution") for s in SEEDS]
        p1 = np.array([r[0] for r in res])
        pk = np.array([r[1] for r in res])
        print(f"{label:22s} {p1.mean():>8.4f} {p1.std(ddof=1)/len(p1)**.5:>6.3f} "
              f"{pk.mean():>8.4f} {pk.std(ddof=1)/len(pk)**.5:>6.3f} "
              f"{pk.mean()-p1.mean():>7.4f}")


def main():
    t0 = time.time()
    print(f"E4  countdown post-training, resolution rule, {E2.GENERATIONS} gens")

    dipec = [("baseline (λ=1)", MECH.baseline)]
    for lam in (1.25, 1.5, 2.0, 4.0, 8.0):
        dipec.append((f"dipec λ={lam}", partial(MECH.dipec_partial, lam=lam)))
    sweep("DiPEC subsample rate", dipec, SEEDS_DIPEC)

    nov = [("baseline", MECH.baseline)]
    for lam in (0.2, 0.4):
        nov.append((f"novelty λ={lam}", partial(MECH.novelty_bonus, lam=lam)))
    nov.append(("RANDOM λ=0.2 (ctrl)", partial(MECH.random_bonus, lam=0.2)))
    nov.append(("RANDOM λ=0.4 (ctrl)", partial(MECH.random_bonus, lam=0.4)))
    sweep("Novelty vs RANDOM control, 8 seeds (sparse reward)", nov, SEEDS_NOV)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
