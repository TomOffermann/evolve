"""E1c -- what is the novelty bonus actually doing?

E1b's control overturned E1's winner: a RANDOM bonus at matched lambda beat every
novelty setting. So the *signal* contributed nothing. The remaining question is what
the bonus does at all, and there is an obvious candidate:

Mixing rank(F) with any second ranking decorrelates the update from fitness. That is
just **reduced selection pressure** -- it slows commitment to the dominant mode. If
so, plain hyperparameters should reproduce the whole effect, and no diversity
mechanism is warranted on this task.

So the decisive arms are not more mechanisms; they are the boring knobs:

    lr x0.7, x0.5   -- commit more slowly
    sigma x1.5      -- explore more per generation

If those match or beat the bonus arms, the honest conclusion is that E1's headline
result was a hyperparameter effect wearing a diversity costume.

10 seeds, deterministic per-seed batching (fixed after E1b: the global torch RNG
made results depend on arm execution order).

Run:  python code/experiments/e1c_decisive.py
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e1_mechanisms as E1
from evolve.diversity import mechanisms as MECH

E1.SEEDS = list(range(10))

ARMS = [
    ("baseline",            MECH.baseline,                        1.0, 1.0),
    ("lr x0.7",             MECH.baseline,                        0.7, 1.0),
    ("lr x0.5",             MECH.baseline,                        0.5, 1.0),
    ("sigma x1.5",          MECH.baseline,                        1.0, 1.5),
    ("sigma x1.5 + lr x0.7", MECH.baseline,                       0.7, 1.5),
    ("RANDOM-bonus λ=0.3",  partial(MECH.random_bonus, lam=0.3),  1.0, 1.0),
    ("novelty λ=0.2",       partial(MECH.novelty_bonus, lam=0.2), 1.0, 1.0),
]


def main():
    print(f"E1c  skew={E1.SKEW}  N={E1.N_POP}  gens={E1.GENERATIONS}  "
          f"seeds={len(E1.SEEDS)}\n")
    rows = []
    for label, fn, lr, sig in ARMS:
        MECH.MECHANISMS["_arm"] = fn
        res = [E1.run("_arm", s, lr_mult=lr, sigma_mult=sig) for s in E1.SEEDS]
        m = np.array([r[0] for r in res])
        w = np.array([r[1] for r in res])
        # standard error, not standard deviation: we are comparing means
        rows.append((label, m.mean(), m.std(ddof=1) / len(m) ** 0.5,
                     w.mean(), w.std(ddof=1) / len(w) ** 0.5))
        print(f"  {label:22s} mean {m.mean():+.4f}  worst {w.mean():+.4f}")

    b = rows[0]
    print(f"\n{'arm':22s} {'mean':>9s} {'sem':>6s} {'Δ':>8s} "
          f"{'worst':>9s} {'sem':>6s} {'Δ':>8s}")
    for label, m, me, w, we in sorted(rows, key=lambda r: -r[3]):
        print(f"{label:22s} {m:>9.4f} {me:>6.3f} {m-b[1]:>+8.4f} "
              f"{w:>9.4f} {we:>6.3f} {w-b[3]:>+8.4f}")
    print("\nsem = standard error of the mean over seeds. A Δ smaller than ~2x the")
    print("combined sem is not a result.")


if __name__ == "__main__":
    main()
