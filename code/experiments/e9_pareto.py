"""E9 -- the last null for Proposal B: is partitioned just learning *less*?

E8 established that partitioned ES beats the best-tuned global arm by +5.0 sem on
pass@16 (0.2319 vs 0.1567), so the "smaller effective step" explanation is dead.
But one null survives, and it is the obvious one:

    Partitioned also has *lower* pass@1 (0.0991 vs 0.1123, ~1.8 sem). A model that
    learns less stays closer to its base, and the base has pass@16 0.211. So "less
    learning" would produce lower pass@1 AND higher pass@16 at the same time --
    exactly the pattern observed.

The test is therefore not a single point but the **whole trade-off curve**: sweep how
far training has progressed and plot pass@1 against pass@16 for both methods.

    If partitioned's curve lies ABOVE global's at matched pass@1  -> structural.
    If both lie on the SAME curve                                 -> it is a knob.

This is the framing the catastrophic-forgetting paper used (arXiv:2601.20861): ES
traced a convex Pareto front while GRPO sat in the top-right corner. Same question,
same method, one level down.

Cheap by construction: the training trajectory *is* the frontier, so one run yields
the whole curve instead of one point per run.

Run:  python code/experiments/e9_pareto.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.core.trainer import TrainConfig, Trainer
from evolve.objectives.countdown_obj import CountdownObjective
from evolve.operators import sampling, sigma, weighting

SEEDS = list(range(5))
GENS = 300
CHECKS = [0, 40, 80, 130, 190, 250, 300]


def trajectory(kind, seed):
    """One run, evaluated at checkpoints -> a whole (pass@1, pass@16) curve."""
    obj = CountdownObjective(n_problems=96, seed=seed)
    smp = (sampling.PartitionedSampler(n_active=1) if kind == "part"
           else sampling.IIDSampler())
    cfg = TrainConfig(n_pop=128, rank=1, sigma=0.005, alpha=0.001,
                      generations=GENS, seed=seed)
    tr = Trainer(obj, smp, weighting.RankWeighting(), cfg,
                 sigma_rule=sigma.ResolutionRule(0.005))
    out = []
    for g in range(GENS + 1):
        if g in CHECKS:
            r = obj.report(ks=(1, 16), seed=seed)
            out.append((g, r["pass@1"], r["pass@16"]))
        if g < GENS:
            tr.step(g)
    tr.close()
    return out


def main():
    print(f"E9  Pareto frontier: is partitioned structural or just slower?")
    print(f"seeds={len(SEEDS)} gens={GENS} checkpoints={CHECKS}")
    print("E8: partitioned 0.2319 vs best global 0.1567 pass@16 (+5.0 sem),")
    print("    but partitioned pass@1 0.0991 vs 0.1123 -- hence this test.\n")

    curves = {}
    for kind, label in [("global", "global (iid)"), ("part", "partitioned n=1")]:
        t0 = time.time()
        runs = [trajectory(kind, s) for s in SEEDS]
        agg = []
        for i, g in enumerate(CHECKS):
            p1 = np.array([r[i][1] for r in runs])
            pk = np.array([r[i][2] for r in runs])
            agg.append((g, p1.mean(), p1.std(ddof=1) / len(p1) ** .5,
                        pk.mean(), pk.std(ddof=1) / len(pk) ** .5))
        curves[label] = agg
        print(f"  {label:18s} {time.time()-t0:6.1f}s")

    for label, agg in curves.items():
        print(f"\n--- {label} ---")
        print(f"{'gen':>5s} {'pass@1':>8s} {'sem':>6s} {'pass@16':>8s} {'sem':>6s}")
        for g, m1, s1, mk, sk in agg:
            print(f"{g:>5d} {m1:>8.4f} {s1:>6.3f} {mk:>8.4f} {sk:>6.3f}")

    # The comparison that matters: interpolate global's pass@16 at partitioned's pass@1
    g_curve = curves["global (iid)"]
    p_curve = curves["partitioned n=1"]
    gx = np.array([r[1] for r in g_curve])
    gy = np.array([r[3] for r in g_curve])
    order = np.argsort(gx)
    print(f"\n{'partitioned point':>22s} {'global pass@16 at same pass@1':>32s} {'Δ':>9s}")
    for g, m1, s1, mk, sk in p_curve[1:]:
        if gx.min() <= m1 <= gx.max():
            gk = float(np.interp(m1, gx[order], gy[order]))
            print(f"  gen{g:>4d} p@1 {m1:.4f} p@16 {mk:.4f}"
                  f"      {gk:>16.4f}   {mk-gk:>+8.4f}")
        else:
            print(f"  gen{g:>4d} p@1 {m1:.4f} p@16 {mk:.4f}"
                  f"      {'(outside global range)':>16s}   {'-':>8s}")

    print("\nPositive Δ at matched pass@1 => partitioned's frontier DOMINATES, and the")
    print("gain is structural. Δ ~ 0 => both methods sit on one curve and partitioned")
    print("is just a slower knob.")


if __name__ == "__main__":
    main()
