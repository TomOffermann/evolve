"""E7 -- the novelty-vs-random comparison at full power, on the framework.

E4 (8 seeds, hand-rolled code) found novelty beating its random control by +2.7 sem
on pass@16, and decision 0003 rests on that. E5 (3 seeds, framework) came out the
other way: novelty 0.1810 +/- 0.022, RANDOM 0.2240 +/- 0.039.

Those are not actually in conflict -- the E5 difference is 0.043 against a combined
sem of 0.045, i.e. under one sem, and RANDOM's sem of 0.039 is enormous. E5 was
built as a *framework regression* and 3 seeds is fine for "does collapse reproduce
and is the ordering vs no-mechanism preserved", both of which passed. It is simply
underpowered to re-decide E4.

But the direction flipped, and this project has already had two conclusions
overturned by taking an underpowered result at face value. So: rerun the comparison
properly on the framework, at E4's seed count, before decision 0003 is treated as
settled.

Run:  python code/experiments/e7_novelty_power.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e5_framework_regression as E5
from evolve.operators import weighting

SEEDS = list(range(8))
GENS = 250

ARMS = [
    ("no mechanism",       lambda: weighting.RankWeighting()),
    ("novelty λ=0.2",      lambda: weighting.NoveltyBonus(lam=0.2)),
    ("novelty λ=0.4",      lambda: weighting.NoveltyBonus(lam=0.4)),
    ("RANDOM λ=0.2 (ctrl)", lambda: weighting.RandomBonus(lam=0.2)),
    ("RANDOM λ=0.4 (ctrl)", lambda: weighting.RandomBonus(lam=0.4)),
]


def main():
    print(f"E7  novelty vs control at full power   seeds={len(SEEDS)} gens={GENS}")
    print("E4 reference (8 seeds, hand-rolled): novelty .2 0.1880 vs RANDOM .2 0.1479\n")
    rows = []
    for label, make in ARMS:
        t0 = time.time()
        res = [E5.run_arm(make(), s, gens=GENS) for s in SEEDS]
        p1 = np.array([r[0] for r in res])
        pk = np.array([r[1] for r in res])
        rows.append((label, p1.mean(), p1.std(ddof=1) / len(p1) ** .5,
                     pk.mean(), pk.std(ddof=1) / len(pk) ** .5))
        print(f"  {label:22s} {time.time()-t0:6.1f}s  pass@1 {p1.mean():.4f}  "
              f"pass@16 {pk.mean():.4f}")

    print(f"\n{'arm':22s} {'pass@1':>8s} {'sem':>6s} {'pass@16':>8s} {'sem':>6s}")
    for label, m1, s1, mk, sk in rows:
        print(f"{label:22s} {m1:>8.4f} {s1:>6.3f} {mk:>8.4f} {sk:>6.3f}")

    def get(name):
        return [r for r in rows if r[0].startswith(name)][0]

    print("\nnovelty vs its matched control:")
    for lam in ("0.2", "0.4"):
        n, c = get(f"novelty λ={lam}"), get(f"RANDOM λ={lam}")
        d = n[3] - c[3]
        se = (n[4] ** 2 + c[4] ** 2) ** 0.5
        print(f"  λ={lam}: pass@16 {n[3]:.4f} vs {c[3]:.4f}  Δ={d:+.4f}  "
              f"{d/se:+.1f} sem   (pass@1 {n[1]:.4f} vs {c[1]:.4f})")
    print("\nA |Δ| under ~2 sem is not a result in either direction.")


if __name__ == "__main__":
    main()
