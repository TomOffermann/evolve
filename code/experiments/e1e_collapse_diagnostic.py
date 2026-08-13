"""E1e -- is there actually any diversity collapse to fix?

Every mechanism in E1 assumed the answer is yes. Nobody checked. If the population's
behavioural diversity never collapses, then every diversity mechanism is solving a
non-problem, and that single fact explains the whole E1 result set at once.

Measured per generation on the free Tier-0 signature:

  eff_rank  -- effective rank of the population's signature matrix. How many
               independent behavioural directions the population spans.
  logdet    -- DvD population volume, log det(K + eps I).
  spread    -- std of raw fitness across the population. Selection pressure.
  worst     -- held-out log-lik on the rarest mode. What we actually want.

Run:  python code/experiments/e1e_collapse_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import e1_mechanisms as E1
from evolve.diversity import mechanisms as MECH
from evolve.diversity.metrics import effective_rank, log_det_volume
from evolve.es import noise
from evolve.es.model import TinyLM, per_example_fitness
from evolve.tasks.grammar import GrammarTask

LOG_EVERY = 25


def trace(mech_name, mech, seed=0):
    task = GrammarTask(n_modes=4, seed=0, skew=E1.SKEW)
    model = TinyLM(task.vocab_size, E1.CTX, device=E1.DEV, seed=seed)
    state = {}
    batch_rng = torch.Generator().manual_seed(50_000 + seed)

    to = lambda a: torch.from_numpy(a).to(E1.DEV)
    Xtr, Ytr, _ = task.contexts(96, E1.CTX, seed=1000 + seed)
    Xte, Yte, Mte = task.contexts(48, E1.CTX, seed=7777)
    Xtr, Ytr, Xte, Yte, Mte = map(to, (Xtr, Ytr, Xte, Yte, Mte))

    out = []
    for gen in range(E1.GENERATIONS):
        lr = E1.ALPHA * (1.0 - 0.7 * gen / E1.GENERATIONS)
        idx = torch.randperm(Xtr.shape[0], generator=batch_rng)[:512]
        Xb, Yb = Xtr[idx], Ytr[idx]
        fac = noise.factors(seed * 100_000 + gen, model.shapes, E1.N_POP, E1.RANK, E1.DEV)
        logits, _ = model.forward_population(Xb, fac, E1.SIGMA, E1.RANK,
                                             capture_signature=False)
        f = per_example_fitness(logits, Yb)
        F = f.mean(dim=1)

        if gen % LOG_EVERY == 0:
            S = MECH._signature(f)
            d = 1.0 - S @ S.T
            _, per_mode = E1.evaluate(model, Xte, Yte, Mte, task.n_modes)
            out.append((gen,
                        float(effective_rank(S)),
                        float(log_det_volume(d)),
                        float(F.std()),
                        per_mode[int(np.argmin(E1.SKEW))]))

        w = mech(F, f, state=state)
        scale = lr / (E1.N_POP * E1.SIGMA * E1.RANK**0.5)
        for nm, (A, B) in fac.items():
            model.W[nm] += scale * torch.einsum("i,imr,ikr->mk", w, A, B)
    return out


def main():
    print(f"E1e  collapse diagnostic   skew={E1.SKEW}  N={E1.N_POP}\n")
    for name, mech in [("baseline", MECH.baseline)]:
        print(f"--- {name} ---")
        print(f"{'gen':>5s} {'eff_rank':>9s} {'logdet':>10s} {'fit_spread':>11s} "
              f"{'worst_mode':>11s}")
        for gen, er, ld, sp, wm in trace(name, mech):
            print(f"{gen:>5d} {er:>9.2f} {ld:>10.2f} {sp:>11.4f} {wm:>11.4f}")

    print("\nIf eff_rank and logdet stay flat while worst_mode degrades, the rare mode")
    print("is being lost for a reason other than population collapse -- and no")
    print("population-diversity mechanism can fix it.")


if __name__ == "__main__":
    main()
