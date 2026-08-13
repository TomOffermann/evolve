"""E0 -- does parameter-space distance mean anything at all?

See research/experiments/E0-does-parameter-distance-mean-anything.md.

Two design points, both learned the hard way while running this:

1. The metric and the target must use *disjoint* data. Otherwise Proposal A Tier 0
   (per-example fitness as a signature) is circular, since the target is also
   per-example fitness. Metrics see probe batch P; the target is measured on
   held-out batch Q. That is also the stronger question: does a signature computed
   on one batch predict behavioural disagreement on data it has never seen?

2. We need a ceiling and a floor. `ORACLE` is the same behavioural target computed
   on P instead of Q -- i.e. the best any P-based metric could possibly do, given
   that behaviour itself does not transfer perfectly between batches. `RANDOM` is a
   shuffled distance matrix. Without these, a rho of 0.35 is uninterpretable.

Run:  python code/experiments/e0_metric_sanity.py [--big]
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.diversity import metrics as M
from evolve.es import eggroll, noise
from evolve.es.model import TinyLM, per_example_fitness
from evolve.tasks.grammar import GrammarTask

torch.manual_seed(0)
DEV = "cpu"  # tiny model; MPS launch overhead dominates at this size

BIG = "--big" in sys.argv
CTX, N_POP, RANK = 6, 256, 1
SIGMA, ALPHA = 0.02, 0.05
EMB, HID = (48, 256) if BIG else (16, 64)
GENERATIONS = 300
CHECKPOINTS = [0, 40, 120, 300]


def spearman(a, b):
    """Rank correlation. (The local scipy install is broken; this is three lines.)"""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def behavioural_target(f):
    """1 - corr(f_i, f_j) on double-centred per-example fitness. The operational
    definition of 'diverse' from P1: distant members should succeed and fail on
    *different* examples, over and above their overall skill."""
    z = M.double_centre(f)
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return 1.0 - z @ z.T


def offdiag(D):
    n = D.shape[0]
    return D[~torch.eye(n, dtype=torch.bool, device=D.device)]


def analyse(model, sketch, XP, YP, XQ, YQ, seed):
    fac = noise.factors(seed, model.shapes, N_POP, RANK, DEV)

    logits_p, own_delta = model.forward_population(XP, fac, SIGMA, RANK)
    total_delta = model.total_output_delta(XP, logits_p)
    f_p = per_example_fitness(logits_p, YP)

    logits_q, _ = model.forward_population(XQ, fac, SIGMA, RANK, capture_signature=False)
    f_q = per_example_fitness(logits_q, YQ)

    target = behavioural_target(f_q)
    basis = sketch.basis(model.shapes)
    g = torch.Generator().manual_seed(seed)

    cands = {
        "-- RANDOM (floor)":                torch.rand(N_POP, N_POP, generator=g),
        "1  param cosine (null hyp.)":      M.param_cosine(fac),
        "4  signed Hamming on delta":       M.signed_hamming(fac),
        "5  per-layer energy profile":      M.layer_energy(fac),
        "6  active subspace (Prop C)":      M.active_subspace(fac, basis),
        "A0 per-example fitness (Prop A)":  M.fitness_vector_signature(f_p),
        "A1 own-layer delta, sketched":     M.delta_activation_signature(own_delta),
        "A2 total output delta, sketched":  M.delta_activation_signature(total_delta),
        "A3 total delta, TASK-PROJECTED":   M.task_projected_signature(total_delta, YP),
        "== CEILING (behaviour on P)":      behavioural_target(f_p),
    }

    t = offdiag(target).numpy()
    rows = []
    for name, D in cands.items():
        if D is None:
            rows.append((name, float("nan"), float("nan")))
            continue
        d = offdiag(D).numpy()
        rows.append((name, spearman(d, t), float(np.std(d))))
    return rows, float(f_q.mean())


def main():
    task = GrammarTask(n_modes=4, seed=0)
    model = TinyLM(task.vocab_size, CTX, emb=EMB, hidden=HID, device=DEV, seed=0)
    sketch = M.SubspaceSketch(k=16, history=8)

    to = lambda a: torch.from_numpy(a).to(DEV)
    Xtr, Ytr, _ = task.contexts(96, CTX, seed=1)
    XP, YP, _ = task.contexts(24, CTX, seed=2)   # probe batch (metrics)
    XQ, YQ, _ = task.contexts(24, CTX, seed=3)   # held-out    (target)
    Xtr, Ytr, XP, YP, XQ, YQ = map(to, (Xtr, Ytr, XP, YP, XQ, YQ))

    n_par = sum(v.numel() for v in model.W.values())
    print(f"vocab={task.vocab_size} ctx={CTX} params={n_par} N={N_POP} "
          f"sigma={SIGMA} rank={RANK}")
    print(f"predicted cos std if concentration holds: 1/sqrt(d) = {1/n_par**0.5:.4f}")
    print(f"probe={XP.shape[0]} ex   held-out={XQ.shape[0]} ex\n")

    for gen in range(GENERATIONS + 1):
        if gen in CHECKPOINTS:
            rows, fq = analyse(model, sketch, XP, YP, XQ, YQ, seed=10_000 + gen)
            print(f"--- gen {gen:<4d} held-out mean log-lik {fq:+.4f} ---")
            print(f"{'metric':34s} {'spearman rho':>13s} {'spread (std)':>13s}")
            for name, rho, sd in rows:
                print(f"{name:34s} {rho:>13.4f} {sd:>13.4f}")
            print()

        if gen == GENERATIONS:
            break
        lr = ALPHA * (1.0 - 0.7 * gen / GENERATIONS)   # decay; without it ES diverges
        before = {k: v.clone() for k, v in model.W.items()}
        idx = torch.randperm(Xtr.shape[0])[:768]
        eggroll.step(model, (Xtr[idx], Ytr[idx]), N_POP, SIGMA, lr, RANK, seed=gen)
        sketch.observe({k: model.W[k] - before[k] for k in model.W})

    print("rho = rank correlation between the metric's pairwise distance and")
    print("behavioural decorrelation on HELD-OUT data. Read every row against")
    print("RANDOM (floor) and CEILING (the same behavioural measure computed on the")
    print("probe batch -- i.e. how well behaviour transfers between batches at all).")
    print("A0 coincides with CEILING by construction: Proposal A Tier 0 *is* that")
    print("statistic. The ceiling is therefore a statement about how much room the")
    print("other metrics are failing to use, not an independent oracle.")


if __name__ == "__main__":
    main()
