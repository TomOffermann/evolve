"""E12 probe -- is there a reproducible part effect at all, and at what (sigma, N)?

The statistic is `part_cos`: fit the part-effect matrix U on one population, fit it
again on a SECOND, INDEPENDENTLY DRAWN population at the same settings, and take the
mean per-part cosine between the two estimates. It answers a question that has to be
settled before any part-level machinery is worth building:

    does u_p mean anything, or is it noise?

Why it matters, and why more samples do not rescue it. Expand the per-example score:

    q_ib = q_b + <g_b, E_i> + 1/2 E_i' H_b E_i + O(sigma^3),   E_i = sigma * (M_i . Z_i)

Condition on member i having perturbed part p and average over the direction Z. The
FIRST-ORDER TERM VANISHES -- E[<g_b, E_i> | part p] = 0, because Z is zero-mean whatever
the mask is. Part identity survives only in the curvature term, so

    signal  = E[q_ib | part p] - q_b  =  1/2 sigma^2 tr(H_b|_p)      O(sigma^2)
    noise   = sd of the first-order term given p = sigma ||g_b|_p||  O(sigma)

    SNR(u_p_hat)  ~  sigma * sqrt(n) * tr(H_b|_p) / ||g_b|_p||

U is a curvature object estimated against gradient noise. Halving sigma costs half the
SNR and needs 4x the members per part to recover. That is why the sigma column moves and
the N column barely does.

Measured 2026-09-19 (countdown, 94k policy, post-SFT, seed 0, exact readout):

     P      N  per-part    sigma   part_cos
     4    256        64    0.005     +0.308
     4   1024       256    0.005     -0.433      <- 4x the members, no better
     4    256        64    0.050     +0.656
     4   1024       256    0.050     +0.760
     4   1024       256    0.200     +0.915
     8   1024       128    0.005     +0.027      <- operating sigma: nothing
     8   1024       128    0.050     +0.629
     8   1024       128    0.200     +0.897

Conclusion: u_p is estimable at sigma ~ 0.05-0.2 with n ~ 10^2 members per part, i.e. at
10-40x the training step size. It cannot be estimated from the training population, which
is what forces the two-timescale probe design in E12-results.md.

Usage:
    python code/experiments/e12_probe_part_replication.py
    python code/experiments/e12_probe_part_replication.py --parts 8 --n-pop 1024 2048 \
        --sigmas 0.02 0.05 0.1 0.2 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "cpu_benchmark"))
sys.path.insert(0, str(_CODE / "experiments"))

from evolve.objectives.countdown_obj import CountdownObjective          # noqa: E402
from operators import FinePartitionedSampler                            # noqa: E402
from e12_additivity import (double_centre, exact_solve_prob, fit_U,     # noqa: E402
                            part_cosines, solution_bank)


def U_for(obj, seqs, msk, ctx, P, n_act, sig, N, tag, lam, device):
    smp = FinePartitionedSampler(n_groups=P, n_active=n_act)
    fac = smp.draw(tag, obj.shapes, N, 1, device)
    M = smp._last_mask.double().cpu()
    Mt = M - M.mean(dim=0, keepdim=True)
    F = double_centre(exact_solve_prob(obj.policy, seqs, msk, ctx, fac, sig, 1, N,
                                       obj.temperature))
    return fit_U(Mt, F, lam), float(F.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n-pop", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.005, 0.05, 0.2])
    ap.add_argument("--n-problems", type=int, default=96)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--sft-steps", type=int, default=1200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output", type=str,
                    default=str(_CODE / "cpu_benchmark" / "results"))
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"

    t0 = time.time()
    obj = CountdownObjective(n_problems=args.n_problems, seed=args.seed,
                             sft_steps=args.sft_steps, device=args.device)
    problems, ctx = obj._problems_for(10_000 + args.seed)
    seqs, msk = solution_bank(obj.task, problems)
    print(f"policy {obj.n_params()} params · device {args.device} · "
          f"max solutions/problem {seqs.shape[1]}")
    print(f"\n{'P':>4} {'N':>6} {'per-part':>9} {'sigma':>8} {'part_cos':>10} "
          f"{'sem':>7} {'|F|':>10}")

    rows = []
    for P in args.parts:
        for N in args.n_pop:
            for sig in args.sigmas:
                vals, norms = [], []
                for r in range(args.repeats):
                    base = (args.seed * 7919) ^ (r * 104729) ^ int(sig * 1e6) ^ N
                    U1, n1 = U_for(obj, seqs, msk, ctx, P, 1, sig, N,
                                   (base ^ 0x1111) & 0x7FFFFFFF, args.lam, args.device)
                    U2, _ = U_for(obj, seqs, msk, ctx, P, 1, sig, N,
                                  (base ^ 0x2222) & 0x7FFFFFFF, args.lam, args.device)
                    vals.append(float(part_cosines(U1, U2).mean()))
                    norms.append(n1)
                v = torch.tensor(vals, dtype=torch.double)
                sem = float(v.std() / v.numel() ** 0.5) if v.numel() > 1 else 0.0
                rows.append({"P": P, "n_pop": N, "per_part": N // P, "sigma": sig,
                             "part_cos": float(v.mean()), "sem": sem,
                             "signal_norm": sum(norms) / len(norms),
                             "repeats": args.repeats})
                print(f"{P:>4} {N:>6} {N // P:>9} {sig:>8.4g} "
                      f"{float(v.mean()):>+10.3f} {sem:>7.3f} "
                      f"{sum(norms) / len(norms):>10.3g}")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    path = out / f"e12_probe_seed{args.seed}.json"
    path.write_text(json.dumps({"experiment": "E12-probe-part-replication",
                                "config": vars(args), "rows": rows,
                                "wall_seconds": time.time() - t0}, indent=2))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")
    print("  part_cos ~ 0        -> no reproducible part effect; u_p is noise")
    print("  part_cos rises with sigma but NOT with N -> second-order, as predicted")
    print("  part_cos > ~0.6     -> u_p is estimable; the part machinery has a basis")


if __name__ == "__main__":
    main()
