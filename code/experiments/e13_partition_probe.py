"""E13-prep -- is there a behavioural partition, and is it different from the default one?

The gate before E13 (learned vs random partition in training). Cheap: no training,
three probe populations per sigma. It answers three questions, in order, and each one
is only worth reading if the previous one passed:

  1. RELIABILITY  -- do per-block signatures replicate across two independent
                     populations on the same problems?  (r_block)
  2. STABILITY    -- does clustering those signatures give the same partition twice?
                     ARI(learned from A, learned from A2), and from held-out problems B.
  3. DIFFERENCE   -- is the learned partition different from the default contiguous one,
                     and does it group blocks that behave alike on HELD-OUT problems
                     better than the default and than random groupings do?

If (3) says learned ~ default, E13 is moot: the tensor layout already IS the functional
structure, and a learned partition cannot beat the default because it is the default.

BLOCKS. 32 explicit row-blocks, not FinePartitionedSampler's auto-allocation -- that
under-allocates (n_groups=8 -> 7 parts, 16 -> 12, 32 -> 25), which silently made E12's
"P=8" a P=7. Here:
    h    192 hidden units  -> 16 blocks of 12
    out  172 vocab rows    -> 12 blocks (~14)
    emb  172 token rows    ->  4 blocks (43)
All three groupings compared below are 8 groups of exactly 4 blocks, so part size is
matched and cannot explain any difference:
    default -- contiguous within a tensor: h 0-3, 4-7, 8-11, 12-15 | out 0-3, 4-7, 8-11 | emb
    random  -- uniformly random balanced assignment (200 draws -> a null distribution)
    learned -- balanced k-means on the probe-A signatures (cosine)

SIGNAL BUDGET. A block has ~1/4 the parameters of a P=8 part. Per the E12 fit the
part signal is ~ d_p sigma^2 tr(H) against noise ~ sqrt(d_p) sigma ||g||, so SNR per member
halves and you need ~4x the members. Defaults: n_active=2 (superposition held at k=2,
E12 Finding 3) and N=4096 -> 256 members per block; sigma 0.1 and 0.2. If r_block
comes out below ~0.5, nothing downstream is interpretable -- raise N before reading on.

Usage:
    python code/experiments/e13_partition_probe.py                  # ~15-20 min CPU
    python code/experiments/e13_partition_probe.py --sigmas 0.2 --n-pop 2048   # quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "cpu_benchmark"))
sys.path.insert(0, str(_CODE / "experiments"))

from evolve.objectives.countdown_obj import CountdownObjective          # noqa: E402
from operators import FinePartitionedSampler                            # noqa: E402
from e12_additivity import (double_centre, exact_solve_prob, fit_U,     # noqa: E402
                            solution_bank)

N_GROUPS, GROUP_SIZE = 8, 4


# ------------------------------------------------------------------ blocks

def make_blocks(shapes):
    """Ordered dict name -> [(matrix, r0, r1)] and the tensor each block lives in."""
    spec = [("h", 16), ("out", 12), ("emb", 4)]
    blocks, tensor = {}, []
    for mat, nb in spec:
        rows = shapes[mat][0]
        edges = np.linspace(0, rows, nb + 1).round().astype(int)
        for j in range(nb):
            blocks[f"{mat}_{j}"] = [(mat, int(edges[j]), int(edges[j + 1]))]
            tensor.append(mat)
    return blocks, tensor


def default_grouping(tensor):
    """Contiguous groups of 4 within each tensor -- the partition E13 must beat."""
    lab, g = np.zeros(len(tensor), int), -1
    counts = {}
    for i, t in enumerate(tensor):
        counts[t] = counts.get(t, 0) + 1
        if (counts[t] - 1) % GROUP_SIZE == 0:
            g += 1
        lab[i] = g
    return lab


def random_grouping(n, rng):
    lab = np.repeat(np.arange(N_GROUPS), GROUP_SIZE)
    rng.shuffle(lab)
    return lab[:n]


# ------------------------------------------------------------- clustering

def balanced_kmeans(X, rng, restarts=20, iters=30):
    """Cosine k-means with every cluster forced to exactly GROUP_SIZE members.

    Balanced so the learned partition has the same part sizes as default and random;
    otherwise a size difference could masquerade as a structure difference."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    best, best_score = None, -np.inf
    for _ in range(restarts):
        C = Xn[rng.choice(len(Xn), N_GROUPS, replace=False)]
        lab = None
        for _ in range(iters):
            sim = Xn @ C.T                                        # (n, G)
            cost = -np.repeat(sim, GROUP_SIZE, axis=1)            # each centroid x4 slots
            r, c = linear_sum_assignment(cost)
            new = c[np.argsort(r)] // GROUP_SIZE
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            C = np.stack([Xn[lab == g].mean(0) for g in range(N_GROUPS)])
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        score = coherence(Xn, lab)
        if score > best_score:
            best, best_score = lab.copy(), score
    return best


def coherence(X, lab):
    """Mean pairwise cosine between blocks in the same group."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    vals = []
    for g in np.unique(lab):
        idx = np.where(lab == g)[0]
        iu = np.triu_indices(len(idx), 1)
        vals.extend(S[np.ix_(idx, idx)][iu])
    return float(np.mean(vals))


def ari(a, b):
    """Adjusted Rand index. 1 = identical partitions, ~0 = chance."""
    from math import comb
    ua, ub = np.unique(a), np.unique(b)
    ct = np.array([[np.sum((a == i) & (b == j)) for j in ub] for i in ua])
    s_ij = sum(comb(int(x), 2) for x in ct.ravel())
    s_a = sum(comb(int(x), 2) for x in ct.sum(1))
    s_b = sum(comb(int(x), 2) for x in ct.sum(0))
    n2 = comb(len(a), 2)
    exp = s_a * s_b / n2
    mx = (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0


# ------------------------------------------------------------------ probe

def probe(obj, blocks, *, sig, n_pop, n_active, tag, gen, lam, device, chunk):
    smp = FinePartitionedSampler(n_groups=len(blocks), n_active=n_active)
    smp._parts = blocks                                  # explicit, not auto-allocated
    fac = smp.draw(tag & 0x7FFFFFFF, obj.shapes, n_pop, 1, device)
    M = smp._last_mask.double().cpu()
    Mt = M - M.mean(0, keepdim=True)
    problems, ctx = obj._problems_for(gen)
    seqs, msk = solution_bank(obj.task, problems)
    F = double_centre(exact_solve_prob(obj.policy, seqs, msk, ctx, fac, sig, 1, n_pop,
                                       obj.temperature, chunk=chunk))
    return fit_U(Mt, F, lam).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--n-pop", type=int, default=4096)
    ap.add_argument("--n-active", type=int, default=2)
    ap.add_argument("--n-problems", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--sft-steps", type=int, default=1200)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output", type=str, default=str(_CODE / "cpu_benchmark" / "results"))
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    t0 = time.time()
    obj = CountdownObjective(n_problems=args.n_problems, seed=args.seed,
                             sft_steps=args.sft_steps, device=args.device)
    blocks, tensor = make_blocks(obj.shapes)
    nb = len(blocks)
    lab_def = default_grouping(tensor)
    print(f"{nb} blocks · {N_GROUPS} groups x {GROUP_SIZE} · n_active={args.n_active} · "
          f"N={args.n_pop} -> {args.n_pop * args.n_active // nb} members/block · seed {args.seed}")

    gen_fit, gen_held = 10_000 + args.seed, 20_000 + args.seed
    rows = []
    for sig in args.sigmas:
        base = (args.seed * 7919) ^ int(sig * 1e6) ^ 0xE13
        kw = dict(sig=sig, n_pop=args.n_pop, n_active=args.n_active, lam=args.lam,
                  device=args.device, chunk=args.chunk)
        UA = probe(obj, blocks, tag=base ^ 0xA, gen=gen_fit, **kw)
        UA2 = probe(obj, blocks, tag=base ^ 0xA2, gen=gen_fit, **kw)
        UB = probe(obj, blocks, tag=base ^ 0xB, gen=gen_held, **kw)

        # 1. reliability
        nA = UA / (np.linalg.norm(UA, axis=1, keepdims=True) + 1e-12)
        nA2 = UA2 / (np.linalg.norm(UA2, axis=1, keepdims=True) + 1e-12)
        r_block = (nA * nA2).sum(1)

        # 2. stability
        rng = np.random.default_rng(args.seed * 31 + int(sig * 1e4))
        L_A = balanced_kmeans(UA, rng)
        L_A2 = balanced_kmeans(UA2, rng)
        L_B = balanced_kmeans(UB, rng)
        rand = [random_grouping(nb, rng) for _ in range(args.n_random)]
        ari_rr = np.mean([ari(rand[i], rand[i + 1]) for i in range(0, len(rand) - 1, 2)])

        # 3. difference and held-out coherence (grouping chosen on A, scored on B)
        coh_rand = np.array([coherence(UB, l) for l in rand])
        coh_L, coh_D = coherence(UB, L_A), coherence(UB, lab_def)
        z = lambda v: (v - coh_rand.mean()) / (coh_rand.std() + 1e-12)
        comp = {g: {t: int(sum(1 for i in np.where(L_A == g)[0] if tensor[i] == t))
                    for t in ("h", "out", "emb")} for g in range(N_GROUPS)}

        row = {"sigma": sig, "r_block_mean": float(r_block.mean()),
               "r_block_min": float(r_block.min()),
               "r_block_by_tensor": {t: float(np.mean([r_block[i] for i in range(nb)
                                                       if tensor[i] == t]))
                                     for t in ("h", "out", "emb")},
               "ari_learned_A_vs_A2": ari(L_A, L_A2), "ari_learned_A_vs_B": ari(L_A, L_B),
               "ari_learned_vs_default": ari(L_A, lab_def), "ari_random_vs_random": float(ari_rr),
               "coh_heldout_learned": coh_L, "coh_heldout_default": coh_D,
               "coh_heldout_random_mean": float(coh_rand.mean()),
               "coh_heldout_random_sd": float(coh_rand.std()),
               "z_learned": float(z(coh_L)), "z_default": float(z(coh_D)),
               "learned_composition": comp, "learned_labels": L_A.tolist(),
               "default_labels": lab_def.tolist(), "block_tensor": tensor}
        rows.append(row)

        print(f"\n=== sigma = {sig:g}  ({time.time() - t0:.0f}s) ===")
        print(f"1 RELIABILITY  r_block mean {row['r_block_mean']:+.3f}  min {row['r_block_min']:+.3f}"
              f"   by tensor " + "  ".join(f"{t} {v:+.2f}" for t, v in row['r_block_by_tensor'].items()))
        print(f"2 STABILITY    ARI(A, A2) {row['ari_learned_A_vs_A2']:+.3f}   "
              f"ARI(A, held-out B) {row['ari_learned_A_vs_B']:+.3f}   "
              f"random-vs-random {row['ari_random_vs_random']:+.3f}")
        print(f"3 DIFFERENCE   ARI(learned, default) {row['ari_learned_vs_default']:+.3f}")
        print(f"  held-out coherence  learned {coh_L:+.3f} (z {row['z_learned']:+.1f})   "
              f"default {coh_D:+.3f} (z {row['z_default']:+.1f})   "
              f"random {coh_rand.mean():+.3f} +/- {coh_rand.std():.3f}")
        print("  learned groups (h/out/emb): " +
              "  ".join(f"{c['h']}/{c['out']}/{c['emb']}" for c in comp.values()))

        # verdict
        if row["r_block_mean"] < 0.5:
            v = "r_block < 0.5: signatures too noisy -- raise --n-pop before reading 2 and 3"
        elif row["ari_learned_A_vs_A2"] < 0.3:
            v = "learned partition is not stable -- it is fitting noise; E13 has nothing to test"
        elif row["ari_learned_vs_default"] > 0.6:
            v = "learned ~ default -- tensor layout already is the functional structure; E13 moot"
        elif row["z_learned"] > 2 and coh_L > coh_D:
            v = "GO: stable, differs from default, and more coherent on held-out problems -> run E13"
        else:
            v = "stable and different, but not more coherent than default on held-out problems"
        print(f"  VERDICT: {v}")
        row["verdict"] = v

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    path = out / f"e13_partition_probe_seed{args.seed}.json"
    path.write_text(json.dumps({"experiment": "E13-partition-probe", "config": vars(args),
                                "rows": rows, "wall_seconds": time.time() - t0}, indent=2))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
