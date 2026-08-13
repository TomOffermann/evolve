"""Seed -> rank-r perturbation factors. Never materialise a dense delta.

EGGROLL constraint C1: population members are *seeds*. `A_i, B_i` are regenerated on
demand from a counter-based RNG and never stored. Everything downstream — including
every diversity metric — must work on the factors.
"""

import torch


def factors(seed: int, shapes, n_pop: int, rank: int, device, dtype=torch.float32):
    """Generate rank-`rank` factors for a whole population, per weight matrix.

    Returns {name: (A, B)} with A: (N, out, r), B: (N, in, r), unit-variance entries.
    Perturbation is E_i = (1/sqrt(r)) A_i B_i^T, matching arXiv:2511.16652.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for name, (m, n) in shapes.items():
        A = torch.randn(n_pop, m, rank, generator=g, dtype=dtype)
        B = torch.randn(n_pop, n, rank, generator=g, dtype=dtype)
        out[name] = (A.to(device), B.to(device))
    return out


def inner_products(A_i, B_i, A_j=None, B_j=None):
    """<E_i, E_j>_F for rank-r factored perturbations, without forming E.

    <A_i B_i^T, A_j B_j^T> = tr(B_i A_i^T A_j B_j^T) = sum((A_i^T A_j) * (B_i^T B_j))

    For r=1 this is just (a_i . a_j)(b_i . b_j) -- note the *product* of two
    near-zero correlations, which is why rank-1 deltas concentrate even harder than
    dense ones. See research/problems/P1.
    """
    if A_j is None:
        A_j, B_j = A_i, B_i
    # (N, N, r, r) would blow up; contract pairwise instead.
    AA = torch.einsum("imr,jms->ijrs", A_i, A_j)
    BB = torch.einsum("inr,jns->ijrs", B_i, B_j)
    return (AA * BB).sum(dim=(-1, -2))
