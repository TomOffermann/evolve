"""E11 -- drift at MATCHED pass@1, not at matched generations.

E10 part 2 found partitioned drifting 2.3x less than global after 250 generations
(||dW|| 5.73 vs 13.10) with much better pass@16 (0.2331 vs 0.1458). That supports
the norm half of arXiv:2601.20861's forgetting diagnosis.

But partitioned also *progresses* less over those 250 generations (pass@1 ~0.104 vs
~0.120), and a model that learns less naturally moves less. Drift-at-equal-
generations is therefore the same confound that E9 had to control for pass@16, one
level down.

The controlled question: **at matched pass@1, does partitioned still drift less?**

    Yes -> partitioning genuinely reaches a given capability with a smaller
           parameter-space excursion, which is the property the forgetting paper
           says matters.
    No  -> the drift difference is just less progress, and only the pass@k result
           stands.

Same trick as E9: log drift at training checkpoints so one run yields a whole
drift-vs-pass@1 curve.

Run:  python code/experiments/e11_drift_controlled.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.core.trainer import TrainConfig, Trainer
from evolve.objectives.countdown_obj import CountdownObjective
from evolve.operators import sampling, sigma, weighting

SEEDS = [0, 1, 2, 3]
GENS = 300
CHECKS = [0, 40, 80, 130, 190, 250, 300]


def trajectory(kind, seed):
    obj = CountdownObjective(n_problems=96, seed=seed)
    base = {k: v.clone() for k, v in obj.policy.W.items()}
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
            drift = sum(float((obj.policy.W[k] - base[k]).norm()) ** 2
                        for k in base) ** 0.5
            out.append((g, r["pass@1"], r["pass@16"], drift))
        if g < GENS:
            tr.step(g)
    tr.close()
    return out


def main():
    print("E11  drift at matched pass@1")
    print(f"seeds={len(SEEDS)} gens={GENS}")
    print("E10 pt2 (uncontrolled): partitioned drift 5.73 vs global 13.10\n")

    curves = {}
    for kind, label in [("global", "global (iid)"), ("part", "partitioned n=1")]:
        t0 = time.time()
        runs = [trajectory(kind, s) for s in SEEDS]
        agg = []
        for i, g in enumerate(CHECKS):
            p1 = np.array([r[i][1] for r in runs])
            pk = np.array([r[i][2] for r in runs])
            dr = np.array([r[i][3] for r in runs])
            agg.append((g, p1.mean(), pk.mean(), dr.mean(),
                        dr.std(ddof=1) / len(dr) ** .5))
        curves[label] = agg
        print(f"  {label:18s} {time.time()-t0:6.1f}s")

    for label, agg in curves.items():
        print(f"\n--- {label} ---")
        print(f"{'gen':>5s} {'pass@1':>8s} {'pass@16':>8s} {'drift':>8s} {'sem':>6s}")
        for g, m1, mk, dr, ds in agg:
            print(f"{g:>5d} {m1:>8.4f} {mk:>8.4f} {dr:>8.3f} {ds:>6.3f}")

    g_curve = curves["global (iid)"]
    gx = np.array([r[1] for r in g_curve])
    gd = np.array([r[3] for r in g_curve])
    order = np.argsort(gx)
    print(f"\n{'partitioned':>28s} {'global drift at same pass@1':>30s} {'ratio':>8s}")
    for g, m1, mk, dr, ds in curves["partitioned n=1"][1:]:
        if gx.min() <= m1 <= gx.max():
            gdi = float(np.interp(m1, gx[order], gd[order]))
            print(f"  gen{g:>4d} p@1 {m1:.4f} drift {dr:>6.3f}"
                  f"      {gdi:>22.3f}   {dr/max(gdi,1e-9):>7.2f}x")
        else:
            print(f"  gen{g:>4d} p@1 {m1:.4f} drift {dr:>6.3f}"
                  f"      {'(outside range)':>22s}   {'-':>8s}")

    print("\nRatio < 1 at matched pass@1 => partitioned reaches the same capability")
    print("with a smaller parameter-space excursion, which is the property the")
    print("forgetting paper identifies. Ratio ~ 1 => the drift gap was just less")
    print("progress, and only the pass@k result stands.")


if __name__ == "__main__":
    main()
