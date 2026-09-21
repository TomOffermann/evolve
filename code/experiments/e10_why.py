"""E10 -- WHY does partitioning work? Two measurements, one prediction.

E8/E9 established the effect. Neither explains it, and the explanation I first
reached for was wrong: partitioning does *not* make the aggregate update sparse
(with 128 members over 3 parts, ~43 hit each part every generation).

The surviving hypothesis is **cross-block credit contamination**. To first order a
global member's fitness is

    F_i ~= F(W) + sigma * sum_q <grad_q, E_i^q>

so the estimator for block p, (1/N sigma) sum_i F_i E_i^p, picks up terms
<grad_q, E_i^q> E_i^p for every other block q. Those are mean-zero but not
variance-zero: global ES cannot tell whether a member scored well because of its
embedding perturbation or its output perturbation.

    global block-p variance      ~ (sum_q ||grad_q||^2) / N
    partitioned block-p variance ~ P * ||grad_p||^2 / N     (only N/P members inform p)

**Falsifiable prediction:** partitioning improves block p's estimate iff
||grad_p||^2 < mean_q ||grad_q||^2 -- it should help below-average blocks and *hurt*
the dominant one. If measured SNR is flat across blocks, or partitioned wins
everywhere including the dominant block, the hypothesis is wrong.

Part 1 -- estimator SNR per block.
    A large-N global population gives a reference direction per block (approximating
    the true gradient). Then compare, over many trials, how well a normal-N global
    estimate and a partitioned estimate align with it. No training involved.

Part 2 -- cumulative drift.
    The other half of arXiv:2601.20861's diagnosis is update *norm*. Train both and
    measure ||W - W_base|| per block. If partitioned drifts less, that supports the
    forgetting link; if it drifts the same or more, the link is dead and the effect
    is something else.

Run:  python code/experiments/e10_why.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.core import noise
from evolve.core.trainer import TrainConfig, Trainer
from evolve.core.types import EvalRequest
from evolve.objectives.countdown_obj import CountdownObjective
from evolve.operators import sampling, sigma, weighting
from evolve.operators.weighting import centred_ranks

DEV = "cpu"
RANK, SIGMA = 1, 0.005
N_REF = 2048          # reference population -- approximates the true gradient
N_EST = 128           # the population size actually used in training
TRIALS = 24
SEEDS = [0, 1, 2]


def block_update(fac, w, rank):
    """{name: dW_p} from factors and weights, without materialising any delta."""
    return noise.accumulate_update(fac, w, rank)


def estimate(obj, smp, n_pop, seed, gen, crn):
    """One block-wise update estimate from a fresh population."""
    fac = smp.draw(seed, obj.shapes, n_pop, RANK, DEV)
    req = EvalRequest(factors=fac, sigma=SIGMA, rank=RANK, crn_seed=crn,
                      generation=gen)
    ev = obj.evaluate(req)
    w = centred_ranks(ev.fitness)
    imp = smp.importance(fac, n_pop)
    if imp is not None:
        w = w * imp
    return block_update(fac, w, RANK)


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


def reliability(ests, name):
    """Mean cosine between INDEPENDENT estimates of the same quantity.

    Reference-free SNR. A large-N reference does not work here: an N-sample ES
    estimate aligns with the true gradient by only ~sqrt(N/d), which at N=128 over
    94k parameters is 0.037 -- so cos(estimate, reference) is a product of two tiny
    numbers and measures mostly the reference's own noise. Split-half reliability
    avoids that: two independent estimates agree only to the extent that both
    contain signal, and the measure is scale-invariant, so samplers with different
    effective sample counts stay comparable."""
    v = [e[name].flatten() for e in ests]
    v = [x / x.norm().clamp_min(1e-12) for x in v]
    M = torch.stack(v) @ torch.stack(v).T
    n = M.shape[0]
    off = M[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean()), float(off.std() / off.numel() ** 0.5)


def part1_snr():
    print("=== Part 1: per-block estimator reliability (reference-free SNR) ===")
    print(f"N={N_EST} per estimate, {TRIALS} independent estimates per sampler\n")
    obj = CountdownObjective(n_problems=96, seed=0)
    iid, part = sampling.IIDSampler(), sampling.PartitionedSampler(n_active=1)
    crn = 12345

    t0 = time.time()
    g_est = [estimate(obj, iid, N_EST, 1000 + t, 0, crn) for t in range(TRIALS)]
    p_est = [estimate(obj, part, N_EST, 5000 + t, 0, crn) for t in range(TRIALS)]
    names = list(g_est[0])
    print(f"{2*TRIALS} estimates in {time.time()-t0:.1f}s")

    # ||grad_p|| from the MEAN of independent estimates (unbiased; a single
    # estimate's norm is inflated by its own noise)
    gnorm = {k: float(torch.stack([e[k] for e in g_est]).mean(0).norm())
             for k in names}
    tot = sum(v**2 for v in gnorm.values())
    mean_sq = tot / len(names)
    print(f"\n{'block':>8s} {'dim':>9s} {'||grad_p||':>11s} {'share':>8s}")
    for k in names:
        print(f"{k:>8s} {g_est[0][k].numel():>9d} {gnorm[k]:>11.4f} "
              f"{gnorm[k]**2/tot:>8.3f}")
    print(f"\nprediction: partitioning HELPS blocks with ||grad_p||^2 below the mean, "
          f"HURTS above")

    print(f"\n{'block':>8s} {'rel(g)':>10s} {'rel(p)':>10s} {'Δ':>9s} {'sem':>7s} "
          f"{'vs mean':>8s} {'pred':>6s} {'match':>8s}")
    hits = 0
    for k in names:
        gm, gs = reliability(g_est, k)
        pm, ps = reliability(p_est, k)
        se = (gs**2 + ps**2) ** 0.5
        rel = gnorm[k]**2 / mean_sq
        pred = "HELP" if rel < 1 else "HURT"
        got = "HELP" if pm > gm else "HURT"
        ok = pred == got
        hits += ok
        print(f"{k:>8s} {gm:>10.4f} {pm:>10.4f} {pm-gm:>+9.4f} {se:>7.4f} "
              f"{rel:>8.2f} {pred:>6s} {'ok' if ok else '**MISS**':>8s}")
    print(f"\nprediction matched on {hits}/{len(names)} blocks")
    print("If the sign of Δ tracks whether ||grad_p||^2 is below the mean, the")
    print("contamination account holds. Uniformly positive Δ means something else")
    print("is going on -- partitioning would then help even the dominant block,")
    print("which cross-block contamination cannot explain.")


def part2_drift(gens=250):
    print(f"\n\n=== Part 2: cumulative drift from base after {gens} generations ===")
    print("arXiv:2601.20861 blames dense HIGH-NORM updates for ES forgetting.\n")
    rows = []
    for kind, label in [("global", "global (iid)"), ("part", "partitioned n=1")]:
        drifts, p16 = [], []
        for seed in SEEDS:
            obj = CountdownObjective(n_problems=96, seed=seed)
            base = {k: v.clone() for k, v in obj.policy.W.items()}
            smp = (sampling.PartitionedSampler(n_active=1) if kind == "part"
                   else sampling.IIDSampler())
            cfg = TrainConfig(n_pop=N_EST, rank=RANK, sigma=SIGMA, alpha=0.001,
                              generations=gens, seed=seed)
            tr = Trainer(obj, smp, weighting.RankWeighting(), cfg,
                         sigma_rule=sigma.ResolutionRule(SIGMA))
            tr.run(); tr.close()
            d = {k: float((obj.policy.W[k] - base[k]).norm()) for k in base}
            rel = {k: d[k] / float(base[k].norm()) for k in base}
            drifts.append((d, rel))
            p16.append(obj.report(ks=(16,), seed=seed)["pass@16"])
        names = list(drifts[0][0])
        agg = {k: (np.mean([x[0][k] for x in drifts]),
                   np.mean([x[1][k] for x in drifts])) for k in names}
        total = np.mean([sum(v**2 for v in x[0].values())**0.5 for x in drifts])
        rows.append((label, agg, total, np.mean(p16)))
        print(f"  {label:18s} total ||ΔW|| {total:7.3f}   pass@16 {np.mean(p16):.4f}")

    names = list(rows[0][1])
    print(f"\n{'block':>8s} " + "".join(f"{r[0]:>22s}" for r in rows))
    for k in names:
        print(f"{k:>8s} " + "".join(
            f"{r[1][k][0]:>10.3f} ({r[1][k][1]*100:>5.1f}%)" for r in rows))
    print(f"{'TOTAL':>8s} " + "".join(f"{r[2]:>10.3f}{'':>10s}" for r in rows))
    print("\nSmaller drift for partitioned supports the forgetting link. Equal or")
    print("larger drift kills it -- the effect would then be something else entirely.")


if __name__ == "__main__":
    part1_snr()
    part2_drift()
