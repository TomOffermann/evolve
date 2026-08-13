"""E1d -- the injection point E1 never tested: selection, not fitness shaping.

P1 names three injection points. E1 tested fitness shaping (novelty, DvD, random)
and sampling (antithetic), and E1b showed the fitness-shaping gains were not
diversity-specific at all. Selection was never tried, and it is where
quality-diversity actually lives.

The argument for expecting something different here: on a skewed task the failure is
not that rare-mode specialists are absent from the population -- it is that they are
**outvoted**. If 3% of the data is mode D, members relatively good at D are a small
minority and their signal is swamped in the fitness mean. A bonus tries to fix that
by paying them more. Niche balancing fixes it by giving each behavioural cluster an
equal vote, which is a much more direct match to the actual failure.

Arms: baseline, the best hyperparameter setting from E1c, the random-bonus control,
and niche balancing at two cluster counts. If niche balancing cannot beat the random
control, fitness-signal-based diversity is dead on this task and the answer is the
boring one.

Run:  python code/experiments/e1d_niche.py
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
    ("baseline",             MECH.baseline,                          1.0, 1.0),
    # The bar to beat, from E1c. Two ordinary hyperparameters, no mechanism.
    ("HP: σx1.5 lrx0.7",     MECH.baseline,                          0.7, 1.5),
    ("RANDOM-bonus λ=0.3",   partial(MECH.random_bonus, lam=0.3),    1.0, 1.0),
    ("niche soft k=8",       partial(MECH.niche_balanced, n_niches=8),  1.0, 1.0),
    ("niche soft k=16",      partial(MECH.niche_balanced, n_niches=16), 1.0, 1.0),
    ("niche hard k=8",       partial(MECH.niche_balanced, n_niches=8, hard=True), 1.0, 1.0),
    # And the honest question: does niching add anything ON TOP of tuned HPs?
    ("niche k=8 + HP",       partial(MECH.niche_balanced, n_niches=8),  0.7, 1.5),
]


def main():
    print(f"E1d  skew={E1.SKEW}  N={E1.N_POP}  gens={E1.GENERATIONS}  "
          f"seeds={len(E1.SEEDS)}\n")
    rows = []
    for label, fn, lr, sig in ARMS:
        MECH.MECHANISMS["_arm"] = fn
        res = [E1.run("_arm", s, lr_mult=lr, sigma_mult=sig) for s in E1.SEEDS]
        m = np.array([r[0] for r in res])
        w = np.array([r[1] for r in res])
        rows.append((label, m.mean(), m.std(ddof=1) / len(m) ** 0.5,
                     w.mean(), w.std(ddof=1) / len(w) ** 0.5))
        print(f"  {label:22s} mean {m.mean():+.4f}  worst {w.mean():+.4f}")

    b = rows[0]
    print(f"\n{'arm':22s} {'mean':>9s} {'sem':>6s} {'Δ':>8s} "
          f"{'worst':>9s} {'sem':>6s} {'Δ':>8s}")
    for label, m, me, w, we in sorted(rows, key=lambda r: -r[3]):
        print(f"{label:22s} {m:>9.4f} {me:>6.3f} {m-b[1]:>+8.4f} "
              f"{w:>9.4f} {we:>6.3f} {w-b[3]:>+8.4f}")
    print("\nsem = standard error over seeds. A Δ smaller than ~2x the combined sem")
    print("is not a result.")


if __name__ == "__main__":
    main()
