"""E1 -- do any of these diversity mechanisms actually improve optimisation?

E0 established that the Tier-0 signature *predicts behaviour*. Necessary, not
sufficient. E1 asks the question that decides whether the direction is worth
anything: routed into the update, does it make the optimiser better?

Task: the grammar with **skewed mode frequencies** [.70, .20, .07, .03]. A greedy
optimiser fits the dominant mode and abandons the rare ones -- the same failure as
diversity collapse in RLVR. On a uniform task with no local optima nothing can help
and every arm ties, which would tell us nothing.

Two numbers per arm, and they are not the same question:
  mean  -- held-out log-likelihood, i.e. raw performance
  worst -- held-out log-likelihood on the *rarest* mode, i.e. what diversity is for

All arms get the same evaluation budget (N members x G generations).

Run:  python code/experiments/e1_mechanisms.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.diversity.mechanisms import MECHANISMS
from evolve.es import noise
from evolve.es.model import TinyLM, per_example_fitness
from evolve.tasks.grammar import GrammarTask

DEV = "cpu"
CTX, N_POP, RANK = 6, 128, 1
SIGMA, ALPHA = 0.02, 0.05
GENERATIONS = 400
SEEDS = [0, 1, 2, 3, 4, 5]
SKEW = [0.70, 0.20, 0.07, 0.03]


def antithetic_factors(seed, shapes, n_pop, rank, device):
    """Mirrored pairs: E and -E. Same N evaluations, half the independent draws.
    -A B^T is A (-B)^T, so mirroring is a sign flip on B."""
    half = noise.factors(seed, shapes, n_pop // 2, rank, device)
    return {k: (torch.cat([A, A]), torch.cat([B, -B])) for k, (A, B) in half.items()}


def evaluate(model, X, Y, M, n_modes):
    """Base-model held-out score: overall mean, and per-mode means."""
    with torch.no_grad():
        logits = model.forward_base(X)
        ll = -torch.nn.functional.cross_entropy(logits, Y, reduction="none")
    per_mode = [ll[M == m].mean().item() for m in range(n_modes)]
    return ll.mean().item(), per_mode


def run(name, seed, lr_mult=1.0, sigma_mult=1.0):
    task = GrammarTask(n_modes=4, seed=0, skew=SKEW)
    model = TinyLM(task.vocab_size, CTX, device=DEV, seed=seed)
    mech = MECHANISMS[name]
    state = {}
    # Dedicated generator for batch selection: identical across arms for a given
    # seed. Using the global torch RNG here made results depend on the order the
    # arms happened to run in -- an uncontrolled variable across the whole table.
    batch_rng = torch.Generator().manual_seed(50_000 + seed)
    sigma = SIGMA * sigma_mult

    to = lambda a: torch.from_numpy(a).to(DEV)
    Xtr, Ytr, _ = task.contexts(96, CTX, seed=1000 + seed)
    Xte, Yte, Mte = task.contexts(48, CTX, seed=7777)
    Xtr, Ytr, Xte, Yte, Mte = map(to, (Xtr, Ytr, Xte, Yte, Mte))

    for gen in range(GENERATIONS):
        lr = ALPHA * lr_mult * (1.0 - 0.7 * gen / GENERATIONS)
        idx = torch.randperm(Xtr.shape[0], generator=batch_rng)[:512]
        Xb, Yb = Xtr[idx], Ytr[idx]

        gs = seed * 100_000 + gen
        fac = (antithetic_factors(gs, model.shapes, N_POP, RANK, DEV)
               if name == "antithetic"
               else noise.factors(gs, model.shapes, N_POP, RANK, DEV))

        logits, _ = model.forward_population(Xb, fac, sigma, RANK,
                                             capture_signature=False)
        f = per_example_fitness(logits, Yb)          # (N, M) -- the free Tier-0 signature
        F = f.mean(dim=1)

        w = mech(F, f, state=state)
        scale = lr / (N_POP * sigma * RANK**0.5)
        for nm, (A, B) in fac.items():
            model.W[nm] += scale * torch.einsum("i,imr,ikr->mk", w, A, B)

    mean, per_mode = evaluate(model, Xte, Yte, Mte, task.n_modes)
    return mean, per_mode[np.argmin(SKEW)], state


def main():
    print(f"E1  task=skewed grammar {SKEW}  N={N_POP}  gens={GENERATIONS}  "
          f"seeds={len(SEEDS)}")
    print("mean  = held-out log-lik (raw performance)")
    print("worst = held-out log-lik on the rarest mode (3% of data)\n")

    rows = []
    for name in MECHANISMS:
        t0 = time.time()
        res = [run(name, s) for s in SEEDS]
        means = np.array([r[0] for r in res])
        worst = np.array([r[1] for r in res])
        rows.append((name, means.mean(), means.std(), worst.mean(), worst.std()))
        print(f"  {name:14s} done in {time.time()-t0:5.1f}s")

    base_m = [r for r in rows if r[0] == "baseline"][0][1]
    base_w = [r for r in rows if r[0] == "baseline"][0][3]

    print(f"\n{'mechanism':15s} {'mean':>9s} {'±':>6s} {'Δ':>8s} "
          f"{'worst':>9s} {'±':>6s} {'Δ':>8s}")
    for name, m, ms, w, ws in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:15s} {m:>9.4f} {ms:>6.3f} {m-base_m:>+8.4f} "
              f"{w:>9.4f} {ws:>6.3f} {w-base_w:>+8.4f}")

    print("\nΔ is versus baseline. A mechanism is only interesting if its Δ exceeds")
    print("the ± spread across seeds -- otherwise it is noise.")


if __name__ == "__main__":
    main()
