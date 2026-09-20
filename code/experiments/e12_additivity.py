"""E12 -- Does the additive part model hold? The gate on Proposal B's decoding claim.

P2 section 2 assumes that for disjoint parts a, b and small sigma

    F(W + E_a + E_b) ~= F(W + E_a) + F(W + E_b) - F(W)

and the whole "one rollout carries P hypotheses" argument rests on it. It has never
been measured. This script measures it, and nothing else.

The object under test is the part x example effect matrix U, fitted by least squares
from things every generation already produces:

    Mt = M - mean(M)                        masks, member-centred          (N, P)
    Ft = (I - 11'/N) F (I - 11'/B)          per-example fitness, double-centred
    U  = (Mt'Mt + lam I)^-1 Mt' Ft                                         (P, B)

Double-centring removes "member i is just good" and "example b is just easy" -- the
two nuisance directions that would otherwise let a useless model look predictive.
Mask centring removes the constant-part degeneracy that makes Mt'Mt singular when
every member activates the same number of parts.

TWO READOUTS, AND WHY THE DEFAULT IS NOT THE TRAINING ONE
---------------------------------------------------------
`--readout binary` is what training uses: one sampled rollout per problem, reward in
{0,1}. `--readout exact` computes, in closed form, the probability that member i emits
a correct expression for problem b:

    q_ib = sum_{s in S_b} prod_t p_i(s_t | ctx_b, s_<t)

S_b is the full set of correct token sequences for problem b -- at most 3! * 3 * 3 = 54
candidates, so it enumerates. q_ib is exactly the quantity the binary reward is a single
Bernoulli sample of, computed with zero sampling noise by teacher forcing.

This is a change of *instrument*, not of objective. Training still optimises the binary
reward; we are asking whether the underlying landscape is additive, and the Bernoulli
noise is not part of the landscape. Measuring the expectation directly is strictly the
better estimator of the thing the question is about.

Run `--calibrate` first. It reports only the reliability ceiling, and on this benchmark
the binary readout has essentially none at the operating sigma -- see the note at the
bottom of this docstring.

SCORING (rule 2 and rule 4, applied to a regression)
-----------------------------------------------------
U has P x B free parameters. Fit in-sample, it will look good no matter what. So the
population is split in half by member:

    fit    U on the train half of (Mt, Ft_a)
    score  cos( Mt_test @ U , Ft_b_test )           out-of-sample
    ceiling cos( Ft_a_test , Ft_b_test )            what ANY model could reach
    score / ceiling  = fraction of REPRODUCIBLE structure the additive model explains

With `--readout exact` the two replicates are identical and the ceiling is 1.0 by
construction; the member split is still what keeps the score honest. With
`--readout binary`, a and b are two independent common-random-number draws, and the
ceiling is the reliability of the instrument itself.

THE CONTROL. Shuffle the rows of Mt inside the train half and refit. Identical rank,
identical regularisation, identical opportunity to overfit -- it just no longer knows
which member perturbed which part. If score_shuffled is not ~0, the pipeline is
measuring something other than what it claims and every other number here is void.

WHAT THE ANSWER MEANS
---------------------
score flat in n_active      -> additivity holds; the regression genuinely decodes
                               several simultaneous perturbations; the swarm-
                               multiplication claim in P2 section 2 is real.
score collapsing in n_active -> cross terms dominate; decoding is dead and everything
                               downstream degrades to per-part means at n_active = 1.

Secondary, free from the same data: the E0 instrument test one level up. Does distance
between part signatures predict held-out behavioural decorrelation between parts?
Held-out means a DIFFERENT PROBLEM SET, not merely different noise, so it is E0's
protocol rather than a restatement of the fit.

MEASURED, 2026-09-19, BEFORE ANY OF THE ABOVE COULD BE RUN
-----------------------------------------------------------
Calibration on countdown, 94k policy, N=256, post-SFT:

    sigma    crn_draws=1   crn_draws=4   crn_draws=16
    0.005      -0.000        -0.006        +0.000
    0.02       -0.000        -0.024        +0.071
    0.05       +0.000        +0.017        +0.207
    0.15       +0.023        +0.167        +0.478

At the sigma E8/E9 actually ran at (0.005), the binary per-example signature has **zero**
reliability: two independent rollouts of the same population agree at r = 0.00. Member
differences in *which problems get solved* are, at that step size, pure sampling noise.

This does not touch E8/E9 -- those measure pass@k after training, not per-generation
signatures. It does mean the Tier-0 signature, validated at rho = 0.76-0.87 in E0 on the
*dense* grammar task, does not survive the move to the sparse one in its sampled form.
Anything indexed by it on this benchmark (Direction B's archive, most obviously) is
indexing noise. Hence `--readout exact`, which is the same signature computed rather
than sampled.

Usage
-----
    python code/experiments/e12_additivity.py --calibrate
    python code/experiments/e12_additivity.py --readout exact --parts 16 \
        --n-active 1 2 4 8 --sigmas 0.0025 0.005 0.01 --n-pop 512 --repeats 3
    python code/experiments/e12_additivity.py --readout exact --train-gens 150
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "cpu_benchmark"))

from evolve.core.trainer import TrainConfig, Trainer                  # noqa: E402
from evolve.core.types import EvalRequest                             # noqa: E402
from evolve.objectives.countdown_obj import CountdownObjective        # noqa: E402
from evolve.operators import sampling, sigma as sigma_ops, weighting  # noqa: E402
from evolve.tasks.countdown import OPS, _eval                         # noqa: E402
from operators import FinePartitionedSampler                          # noqa: E402


# --------------------------------------------------------------------- algebra

def double_centre(F: torch.Tensor) -> torch.Tensor:
    """(I - 11'/N) F (I - 11'/B). The projections commute, so sequential is exact."""
    F = F - F.mean(dim=0, keepdim=True)
    return F - F.mean(dim=1, keepdim=True)


def fit_U(M: torch.Tensor, F: torch.Tensor, lam: float) -> torch.Tensor:
    G = M.T @ M + lam * torch.eye(M.shape[1], dtype=M.dtype)
    return torch.linalg.solve(G, M.T @ F)


def cos(X: torch.Tensor, Y: torch.Tensor) -> float:
    return float((X * Y).sum() / (X.norm() * Y.norm() + 1e-12))


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 3:
        return float("nan")
    rx = torch.argsort(torch.argsort(x)).double()
    ry = torch.argsort(torch.argsort(y)).double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))


# ------------------------------------------------------- the noiseless readout

def solution_bank(task, problems):
    """(B, S, 5) correct token sequences per problem, plus a (B, S) validity mask.

    Countdown's output is `a op b op c` with (a,b,c) a permutation of the three given
    numbers: 3! * 3 * 3 = 54 candidates, so the correct set enumerates exactly. This is
    what makes an exact per-example solve probability available at all."""
    banks = []
    for nums, target in problems:
        seqs = [[task.tok_num[a], task.tok_op[o1], task.tok_num[b],
                 task.tok_op[o2], task.tok_num[c]]
                for a, b, c in itertools.permutations(nums)
                for o1 in OPS for o2 in OPS
                if _eval(a, o1, b, o2, c) == target]
        banks.append(seqs)
    S = max(len(b) for b in banks)
    out = torch.zeros(len(banks), S, 5, dtype=torch.long)
    msk = torch.zeros(len(banks), S, dtype=torch.bool)
    for i, seqs in enumerate(banks):
        for j, s in enumerate(seqs):
            out[i, j] = torch.tensor(s)
            msk[i, j] = True
    return out, msk


@torch.no_grad()
def exact_solve_prob(policy, seqs, msk, ctx, factors, sig, rank, n_pop,
                     temperature=1.0, chunk=96):
    """(N, B) exact P[member i emits a correct expression for problem b].

    Teacher-forced: five forward passes over the candidate bank, no sampling anywhere,
    so two evaluations of the same population agree exactly. That is the entire point --
    the binary readout's reliability at sigma = 0.005 is 0.00."""
    dev = policy.W["emb"].device
    B, S, L = seqs.shape
    seqs, msk = seqs.to(dev), msk.to(dev)
    flat = seqs.reshape(B * S, L)
    ctx_rep = ctx.to(dev).repeat_interleave(S, dim=0)
    # fp32 on device: float64 runs at 1/32 rate on a T4 and the log-probs do not need it.
    total = torch.zeros(n_pop, B * S, dtype=torch.float32, device=dev)

    for lo in range(0, B * S, chunk):
        hi = min(lo + chunk, B * S)
        seq = torch.zeros(n_pop, hi - lo, policy.ctx, dtype=torch.long, device=dev)
        seq[:, :, : ctx_rep.shape[1]] = ctx_rep[lo:hi].unsqueeze(0)
        start = ctx_rep.shape[1]
        acc = torch.zeros(n_pop, hi - lo, dtype=torch.float32, device=dev)
        for t in range(L):
            lp = torch.log_softmax(
                policy.logits_pop(seq, factors, sig, rank) / temperature, dim=-1)
            tgt = flat[lo:hi, t].view(1, -1, 1).expand(n_pop, -1, 1)
            acc += torch.gather(lp, 2, tgt).squeeze(-1)
            seq[:, :, start + t] = flat[lo:hi, t].view(1, -1)
        total[:, lo:hi] = acc
    q = (total.exp().reshape(n_pop, B, S) * msk.view(1, B, S)).sum(-1)
    # The regression itself is tiny (N x B) and stays in float64 on the CPU.
    return q.double().cpu()


# ----------------------------------------------------------------- measurement

def readout(obj, kind, factors, sig, rank, n_pop, generation, crn_base, crn_draws,
            bank_cache):
    """(N, B) per-example signature under the chosen instrument."""
    P, C = obj._problems_for(generation)
    if kind == "exact":
        if generation not in bank_cache:
            bank_cache[generation] = solution_bank(obj.task, P)
        seqs, msk = bank_cache[generation]
        return exact_solve_prob(obj.policy, seqs, msk, C, factors, sig, rank,
                                n_pop, obj.temperature).double()
    acc = 0.0
    for i in range(crn_draws):
        req = EvalRequest(factors=factors, sigma=sig, rank=rank,
                          crn_seed=int(crn_base + i), generation=int(generation),
                          chunk_id=0, n_chunks=1)
        acc = acc + obj.evaluate(req).per_example.double().cpu()
    return acc / crn_draws


def _population(obj, *, P, n_active, sig, n_pop, rank, tag, gen, lam, kind,
                crn_draws, bank_cache, device="cpu"):
    """Draw a population at a given n_active and return (centred masks, centred F)."""
    smp = FinePartitionedSampler(n_groups=P, n_active=n_active)
    ds = tag & 0x7FFFFFFF
    fac = smp.draw(ds, obj.shapes, n_pop, rank, device, index_offset=0)
    M = smp._last_mask.double().cpu()
    Mt = M - M.mean(dim=0, keepdim=True)
    F = double_centre(readout(obj, kind, fac, sig, rank, n_pop, gen,
                              ds ^ 0xA1, crn_draws, bank_cache))
    return Mt, F, ds, fac


def part_cosines(U, V):
    """Per-part agreement between two estimates of the part-effect matrix."""
    A = torch.nn.functional.normalize(U, dim=1)
    B = torch.nn.functional.normalize(V, dim=1)
    return (A * B).sum(dim=1)


def measure(obj, *, P, sig, n_actives, n_pop, rank, seed, rep, gen_fit, gen_held,
            lam, kind, crn_draws, bank_cache, device="cpu", n_shuffles=32):
    """One (P, sigma, repeat) cell. Returns one row per n_active.

    The statistic is a direct test of the P2 section 2 equation. Fit the part-effect
    matrix U once on a reference population where every member perturbs exactly ONE
    part -- there u_p is measured in isolation, by construction. Then fit it again on
    a population at n_active = k, where every measurement is contaminated by whatever
    the other k-1 active parts were doing. Superposition holds iff the two agree:

        part_cos(k) = mean_p cos( u_p measured alone , u_p measured in company )

    k = 1 is included with an INDEPENDENTLY drawn population, so part_cos(1) is not
    1.0 by construction -- it is the replication ceiling of the estimate itself, and
    every larger k is read against it. That is the ceiling this experiment needs, and
    it costs one extra population.

    Reported alongside, because it answers a different question people will ask:
      in_regime  -- how much of the signature the mask explains at all, out of sample.
                    Low is expected and is not a failure of additivity: it is the
                    continuous half of the genotype (which direction inside the part)
                    doing the work the discrete half cannot.
      transfer   -- the operational version: does the isolated model predict combined
                    behaviour? transfer / in_regime is additivity, measured predictively.
    """
    base = (seed * 1_000_003) ^ (0xE12 * (rep + 1)) ^ int(sig * 1e7)
    Mr, Fr, ds_r, fac_r = _population(obj, P=P, n_active=1, sig=sig, n_pop=n_pop, rank=rank,
                            tag=base ^ 0x5E1F, gen=gen_fit, lam=lam, kind=kind,
                            crn_draws=crn_draws, bank_cache=bank_cache, device=device)
    U1 = fit_U(Mr, Fr, lam)                       # u_p measured in isolation
    P_eff = U1.shape[0]

    # Held-out problems AND an independent population -> E0's protocol one level up.
    # Reusing the reference population here leaks the within-part directions, which
    # makes rho_parts look like ~0.65 even when the part effect itself is pure noise.
    # does distance between part signatures predict behavioural decorrelation on
    # problems the signatures were never fitted on?
    Mr2, _, _, fac_r2 = _population(obj, P=P, n_active=1, sig=sig, n_pop=n_pop,
                                    rank=rank, tag=base ^ 0x7A2D, gen=gen_fit,
                                    lam=lam, kind=kind, crn_draws=crn_draws,
                                    bank_cache=bank_cache)
    Fh = double_centre(readout(obj, kind, fac_r2, sig, rank, n_pop, gen_held,
                               (base ^ 0x7A2D) ^ 0xC3, crn_draws, bank_cache))
    Uh = fit_U(Mr2, Fh, lam)
    rho_parts = float("nan")
    if P_eff >= 6:
        A = torch.nn.functional.normalize(U1, dim=1)
        H = torch.nn.functional.normalize(Uh, dim=1)
        iu = torch.triu_indices(P_eff, P_eff, offset=1)
        rho_parts = spearman(torch.cdist(A, A)[iu[0], iu[1]],
                             (1.0 - H @ H.T)[iu[0], iu[1]])

    rows = []
    for k in n_actives:
        if k >= P_eff:
            continue
        Mk, Fk, ds, _ = _population(obj, P=P, n_active=k, sig=sig, n_pop=n_pop,
                                 rank=rank, tag=base ^ (0xB00 * k), gen=gen_fit,
                                 lam=lam, kind=kind, crn_draws=crn_draws,
                                 bank_cache=bank_cache, device=device)
        g = torch.Generator().manual_seed(ds)
        perm = torch.randperm(n_pop, generator=g)
        tr, te = perm[: n_pop // 2], perm[n_pop // 2:]

        Uk = fit_U(Mk[tr], Fk[tr], lam)
        in_regime = cos(Mk[te] @ Uk, Fk[te])
        transfer = cos(Mk[te] @ U1, Fk[te])

        Uk_full = fit_U(Mk, Fk, lam)
        pc = part_cosines(U1, Uk_full)
        # One permutation of P ~ 8 labels is a very noisy control -- a random
        # permutation leaves parts fixed often enough that a single draw reads
        # +0.2 on pure chance. Average it.
        pc_shuf = torch.stack([
            part_cosines(U1, Uk_full[torch.randperm(P_eff, generator=g)])
            for _ in range(n_shuffles)]).mean()

        rows.append({
            "P": P_eff, "n_active": k, "sigma": sig, "repeat": rep, "readout": kind,
            "part_cos": float(pc.mean()),
            "part_cos_sem": float(pc.std() / P_eff ** 0.5),
            "part_cos_shuffled": float(pc_shuf),
            "in_regime": in_regime,
            "transfer": transfer,
            "transfer_ratio": transfer / in_regime if abs(in_regime) > 1e-3 else float("nan"),
            "rho_parts_vs_heldout": rho_parts,
            "signal_norm": float(Fk.norm()),
            "mask_rank": int(torch.linalg.matrix_rank(Mk).item()),
        })
    return rows


# ------------------------------------------------------------------------ main

def mean_sem(vals):
    v = torch.tensor([x for x in vals if x == x], dtype=torch.double)
    if v.numel() == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std() / v.numel() ** 0.5) if v.numel() > 1 else 0.0)


def calibrate(obj, args, bank_cache):
    """Reliability of the instrument, before trusting anything measured with it."""
    print("\nreliability ceiling -- two independent evaluations of the SAME population")
    print("anything below ~0.05 means the readout carries no reproducible signal\n")
    smp = FinePartitionedSampler(n_groups=8, n_active=1)
    fac = smp.draw(123, obj.shapes, args.n_pop, args.rank, args.device)
    gen = 10_000 + args.seed
    rows = []
    for sig in args.sigmas:
        for m in args.crn_sweep:
            A = double_centre(readout(obj, "binary", fac, sig, args.rank, args.n_pop,
                                      gen, 1000, m, bank_cache))
            B = double_centre(readout(obj, "binary", fac, sig, args.rank, args.n_pop,
                                      gen, 5000, m, bank_cache))
            r = cos(A, B)
            rows.append({"readout": "binary", "sigma": sig, "crn_draws": m, "ceiling": r})
            print(f"  binary  sigma={sig:<8.4g} crn_draws={m:<4d} ceiling={r:+.4f}")
        E = double_centre(readout(obj, "exact", fac, sig, args.rank, args.n_pop,
                                  gen, 0, 1, bank_cache))
        rows.append({"readout": "exact", "sigma": sig, "crn_draws": 0,
                     "ceiling": 1.0, "signal_norm": float(E.norm())})
        print(f"  exact   sigma={sig:<8.4g} ceiling=1.0000 (noiseless)  "
              f"|signal|={float(E.norm()):.4g}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", choices=["exact", "binary"], default="exact")
    ap.add_argument("--parts", type=int, nargs="+", default=[16])
    ap.add_argument("--n-active", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0025, 0.005, 0.01])
    ap.add_argument("--n-pop", type=int, default=512)
    ap.add_argument("--n-problems", type=int, default=96)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--crn-draws", type=int, default=1)
    ap.add_argument("--crn-sweep", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--train-gens", type=int, default=0)
    ap.add_argument("--sft-steps", type=int, default=1200)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu or cuda. cuda is the fast path on Colab; CPU is the "
                         "verified one -- check --calibrate agrees before trusting a "
                         "GPU sweep.")
    ap.add_argument("--output", type=str, default=str(_CODE / "cpu_benchmark" / "results"))
    args = ap.parse_args()

    if args.quick:
        args.parts, args.n_active = [8], [1, 2, 4]
        args.sigmas, args.n_pop, args.repeats = [0.005], 128, 1

    t0 = time.time()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"
    obj = CountdownObjective(n_problems=args.n_problems, seed=args.seed,
                             sft_steps=args.sft_steps, device=args.device)
    print(f"policy: {obj.n_params()} params, shapes {dict(obj.shapes)}")
    bank_cache: dict = {}

    if args.train_gens:
        tr = Trainer(obj, sampler=sampling.PartitionedSampler(n_active=1),
                     weighting=weighting.RankWeighting(),
                     config=TrainConfig(n_pop=128, rank=args.rank, sigma=0.005,
                                        alpha=0.0005, generations=args.train_gens,
                                        seed=args.seed),
                     sigma_rule=sigma_ops.ResolutionRule(sigma=0.005))
        tr.run(args.train_gens)
        tr.close()
        print(f"advanced {args.train_gens} generations; {obj.report()}")

    if args.calibrate:
        rows = calibrate(obj, args, bank_cache)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        (out / f"e12_calibration_seed{args.seed}.json").write_text(
            json.dumps({"experiment": "E12-calibration", "config": vars(args),
                        "rows": rows}, indent=2))
        print(f"\n({time.time() - t0:.0f}s)")
        return

    gen_fit, gen_held = 10_000 + args.seed, 20_000 + args.seed
    rows = []
    for P in args.parts:
        for sig in args.sigmas:
            cells = []
            for r in range(args.repeats):
                cells.extend(measure(obj, P=P, sig=sig, n_actives=args.n_active,
                                     n_pop=args.n_pop, rank=args.rank,
                                     seed=args.seed, rep=r, gen_fit=gen_fit,
                                     gen_held=gen_held, lam=args.lam,
                                     kind=args.readout, crn_draws=args.crn_draws,
                                     bank_cache=bank_cache, device=args.device))
            rows.extend(cells)
            print(f"\n  P={P}  sigma={sig:g}  readout={args.readout}  "
                  f"N={args.n_pop}  repeats={args.repeats}")
            print(f"  {'n_act':>6} {'part_cos':>10} {'shuffled':>10} "
                  f"{'additivity':>11} {'in_regime':>10} {'transfer':>9} {'rho_parts':>10}")
            ceiling = None
            for na in args.n_active:
                sub = [c for c in cells if c["n_active"] == na]
                if not sub:
                    continue
                pc, pcs = mean_sem([c["part_cos"] for c in sub])
                shf, _ = mean_sem([c["part_cos_shuffled"] for c in sub])
                ir, _ = mean_sem([c["in_regime"] for c in sub])
                tf, _ = mean_sem([c["transfer"] for c in sub])
                rp, _ = mean_sem([c["rho_parts_vs_heldout"] for c in sub])
                if ceiling is None:
                    ceiling = pc          # n_active=1, independent population
                add = pc / ceiling if ceiling and abs(ceiling) > 1e-6 else float("nan")
                print(f"  {na:>6} {pc:>+10.3f} {shf:>+10.3f} {add:>+11.3f} "
                      f"{ir:>+10.3f} {tf:>+9.3f} {rp:>+10.3f}")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    path = out / f"e12_additivity_{args.readout}_seed{args.seed}_g{args.train_gens}.json"
    path.write_text(json.dumps({"experiment": "E12-additivity", "config": vars(args),
                                "n_params": obj.n_params(), "rows": rows,
                                "wall_seconds": time.time() - t0}, indent=2))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")
    print("  additivity ~ 1 across n_active -> superposition holds, decoding is real")
    print("  additivity falling in n_active -> cross terms dominate, decoding dead")
    print("  shuffled far from 0            -> pipeline broken, ignore the rest")
    print("  in_regime is NOT additivity: it is how much the mask explains at all")


if __name__ == "__main__":
    main()
