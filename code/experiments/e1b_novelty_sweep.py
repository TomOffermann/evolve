"""E1b -- is the novelty win real, and how sharp is it?

E1 picked `novelty` (fixed-lambda k-NN bonus on the Tier-0 signature). Three things
have to be true before committing to it:

  1. It beats a RANDOM bonus of the same weight. Otherwise "novelty helps" is
     unfalsifiable: any mechanism that partly decouples the update from fitness
     injects exploration, and that alone can rescue a rare mode.
  2. It is not a knife-edge in lambda.
  3. It does not *hurt* on a uniform task, where there is nothing for diversity to
     buy. A mechanism that only ever helps is suspicious; one that is neutral when
     it should be neutral is trustworthy.

Run:  python code/experiments/e1b_novelty_sweep.py
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e1_mechanisms as E1
from evolve.diversity import mechanisms as MECH


def sweep(arms, skew, label):
    print(f"\n### {label}   skew={skew}")
    E1.SKEW = skew
    rows = []
    for name, fn in arms.items():
        MECH.MECHANISMS[name] = fn
        res = [E1.run(name, s) for s in E1.SEEDS]
        m = np.array([r[0] for r in res])
        w = np.array([r[1] for r in res])
        rows.append((name, m.mean(), m.std(), w.mean(), w.std()))
        print(f"  {name:22s} mean {m.mean():+.4f} ±{m.std():.3f}   "
              f"worst {w.mean():+.4f} ±{w.std():.3f}")
    b = [r for r in rows if r[0] == "baseline"][0]
    print(f"\n  {'arm':22s} {'Δ mean':>9s} {'Δ worst':>9s}")
    for name, m, ms, w, ws in rows:
        print(f"  {name:22s} {m-b[1]:>+9.4f} {w-b[3]:>+9.4f}")
    return rows


def main():
    arms = {"baseline": MECH.baseline}
    for lam in (0.1, 0.2, 0.3, 0.5, 0.7):
        arms[f"novelty λ={lam}"] = partial(MECH.novelty_bonus, lam=lam)
    arms["RANDOM-bonus λ=0.3"] = partial(MECH.random_bonus, lam=0.3)
    arms["RANDOM-bonus λ=0.5"] = partial(MECH.random_bonus, lam=0.5)
    sweep(arms, [0.70, 0.20, 0.07, 0.03], "skewed task (diversity should pay)")

    control = {
        "baseline": MECH.baseline,
        "novelty λ=0.3": partial(MECH.novelty_bonus, lam=0.3),
        "RANDOM-bonus λ=0.3": partial(MECH.random_bonus, lam=0.3),
    }
    sweep(control, [0.25, 0.25, 0.25, 0.25], "uniform task (nothing to buy)")


if __name__ == "__main__":
    main()
