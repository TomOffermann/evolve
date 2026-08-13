"""The EGGROLL update loop.

    dW = (alpha / (N sigma)) * sum_i F~_i * A_i B_i^T ,   F~_i = F_i - mean(F)

Individual perturbations are rank-r; the *sum* of N of them has rank up to
min(N r, m, n), so the aggregate update is full-rank. That is the difference
between this and running ES on a LoRA adapter.
"""

import torch

from . import noise
from .model import per_example_fitness


def rank_normalise(F):
    """Centred rank transform with **tied ranks averaged**.

    The naive version (argsort -> distinct ranks) is fine for dense fitness but
    actively harmful under sparse binary reward, where many members score exactly
    the same. It breaks those ties by argsort order, which is arbitrary, and so
    injects pure noise into the update in proportion to how sparse the reward is.
    Averaging tied ranks makes equal fitness produce equal weight."""
    N = F.numel()
    order = F.argsort()
    ranks = torch.empty_like(F)
    ranks[order] = torch.arange(N, dtype=F.dtype, device=F.device)

    sortedF = F[order]
    # Average ranks within each run of equal values.
    uniq, inv, counts = torch.unique(sortedF, return_inverse=True, return_counts=True)
    sums = torch.zeros(uniq.numel(), dtype=F.dtype, device=F.device)
    sums.index_add_(0, inv, torch.arange(N, dtype=F.dtype, device=F.device))
    ranks[order] = (sums / counts)[inv]
    return ranks / (N - 1) - 0.5


def step(model, task_batch, n_pop, sigma, alpha, rank, seed, shape=True):
    """One generation. Returns (mean_fitness, per_example_fitness, factors, signature)."""
    X, Y = task_batch
    fac = noise.factors(seed, model.shapes, n_pop, rank, model.device)
    logits, sig = model.forward_population(X, fac, sigma, rank)
    f = per_example_fitness(logits, Y)          # (N, B)
    F = f.mean(dim=1)                            # (N,)

    w = rank_normalise(F) if shape else (F - F.mean())
    scale = alpha / (n_pop * sigma)
    for name, (A, B) in fac.items():
        # sum_i w_i A_i B_i^T, computed without forming per-member deltas
        dW = torch.einsum("i,imr,ikr->mk", w, A, B)
        model.W[name] += (scale / rank**0.5) * dW

    return F, f, fac, sig
