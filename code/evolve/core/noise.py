"""Seed -> low-rank factors. Chunked, device-aware, never stored.

EGGROLL's constraint C1: population members are *seeds*. `A_i, B_i` are regenerated
on demand and never persist. That is what makes a population of 10^6 possible, and
it is also what makes the two-pass trainer loop free (see `trainer.py`): we can
evaluate the whole population, compute weights over all of it, then regenerate the
same factors to accumulate the update, without storing anything.

Determinism contract: `draw(seed, ...)` must return bit-identical factors for the
same `(seed, shapes, n_pop, rank)` on the same device. The trainer relies on it.
Chunking is done by *deriving* a per-chunk seed, so chunk boundaries do not change
the sequence a member sees.
"""

from __future__ import annotations

import torch

from .types import Factors, Shapes


def chunk_seed(seed: int, chunk_id: int) -> int:
    """Derive a stable per-chunk seed. Splitmix-style mix so that nearby (seed,
    chunk) pairs do not produce correlated streams."""
    x = (seed * 0x9E3779B97F4A7C15 + chunk_id * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    return x & 0x7FFFFFFF


def member_seed(seed: int, index: int) -> int:
    """Noise identity of a member is its **global index**, never its position in a
    chunk. This is what makes chunking a pure memory/parallelism decision with no
    effect on results -- verified by `test_chunking_is_exact`.

    Keying on chunk id instead (the obvious first implementation) means changing
    `chunk_size` silently changes which perturbation each member draws, so a large
    sharded run cannot be verified against a small local one. Nothing in the fitness
    curves would reveal it."""
    return chunk_seed(seed ^ 0xA5A5A5A5, index)


def draw_factors(seed: int, shapes: Shapes, n_pop: int, rank: int, device,
                 dtype=torch.float32, index_offset: int = 0) -> Factors:
    """{name: (A, B)} with A: (N, out, r), B: (N, in, r), unit-variance entries.

    Perturbation is E_i = (1/sqrt(r)) A_i B_i^T (arXiv:2511.16652). The 1/sqrt(r)
    lives at the point of use, not here, so callers can reason about raw factors.

    `index_offset` is the member's global index, so a chunk covering [lo, hi) draws
    exactly what the unchunked run would have drawn for those members.

    Per-member seeding costs one generator per member. At the scales here that is
    negligible against the forward pass; the production path for N ~ 10^6 is a
    counter-based RNG (EGGROLL uses `jax.random.fold_in`), which gives the same
    index-keyed property without the per-member object."""
    out: Factors = {}
    sizes = {name: (m * rank, n * rank) for name, (m, n) in shapes.items()}
    total = sum(a + b for a, b in sizes.values())

    flat = torch.empty(n_pop, total, dtype=dtype)
    for j in range(n_pop):
        g = torch.Generator(device="cpu").manual_seed(member_seed(seed, index_offset + j))
        flat[j] = torch.randn(total, generator=g, dtype=dtype)

    pos = 0
    for name, (m, n) in shapes.items():
        na, nb = sizes[name]
        A = flat[:, pos:pos + na].reshape(n_pop, m, rank)
        pos += na
        B = flat[:, pos:pos + nb].reshape(n_pop, n, rank)
        pos += nb
        out[name] = (A.to(device), B.to(device))
    return out


def slice_factors(factors: Factors, lo: int, hi: int) -> Factors:
    return {k: (A[lo:hi], B[lo:hi]) for k, (A, B) in factors.items()}


def accumulate_update(factors: Factors, weights: torch.Tensor, rank: int,
                      into: dict[str, torch.Tensor] | None = None
                      ) -> dict[str, torch.Tensor]:
    """dW += sum_i w_i A_i B_i^T, computed without materialising any per-member delta.

    Individual perturbations are rank-r; the *sum* over N of them has rank up to
    min(N r, m, n), which is why EGGROLL gets full-rank updates from rank-1 noise.
    Accumulating in place across chunks is what lets N exceed memory."""
    into = {} if into is None else into
    s = rank ** -0.5
    for name, (A, B) in factors.items():
        d = torch.einsum("i,imr,ikr->mk", weights.to(A.dtype), A, B) * s
        into[name] = d if name not in into else into[name] + d
    return into


def inner_products(A_i, B_i, A_j=None, B_j=None):
    """<E_i, E_j>_F for factored perturbations, without forming E.

    <A_i B_i^T, A_j B_j^T> = sum((A_i^T A_j) * (B_i^T B_j)).

    For r=1 this is (a_i . a_j)(b_i . b_j) -- a *product* of two near-zero
    correlations, which is why rank-1 deltas concentrate even harder than dense
    ones. E0 measured the resulting spread at exactly 1/sqrt(d)."""
    if A_j is None:
        A_j, B_j = A_i, B_i
    AA = torch.einsum("imr,jms->ijrs", A_i, A_j)
    BB = torch.einsum("inr,jns->ijrs", B_i, B_j)
    return (AA * BB).sum(dim=(-1, -2))
