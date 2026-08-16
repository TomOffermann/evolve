"""Seed-based factor generation for HuggingFace models.

Extends evolve.core.noise with per-layer-per-member seeding. The existing
framework generates all factors for all shapes at once; at 1.5B parameters
(~200 nn.Linear layers) that would require materialising ~48GB of factors
for N=64. Instead, factors are generated one layer at a time, either in
forward hooks (pass 1) or during accumulation (pass 2).

The determinism contract is unchanged: same (base_seed, member_idx, layer_idx)
always produces the same (A, B). This is what makes the two-pass loop free.
"""

import torch

# Reuse the proven seed-mixing from the existing framework
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolve.core.noise import chunk_seed, member_seed


def layer_member_seed(base_seed: int, member_idx: int, layer_idx: int) -> int:
    """Deterministic seed for one member at one layer.

    Composed from member_seed (which keys on global index) and layer_idx,
    so the same member at the same layer always gets the same factors
    regardless of chunking or evaluation order.
    """
    return chunk_seed(member_seed(base_seed, member_idx), layer_idx)


def draw_layer_factors(base_seed: int, member_idx: int, layer_idx: int,
                       out_features: int, in_features: int, rank: int,
                       device, dtype=torch.float32):
    """Generate (A, B) for one member at one layer.

    A: (out_features, rank), B: (in_features, rank), unit-variance entries.
    The 1/sqrt(rank) scaling lives at the point of use (the hook), not here.
    """
    seed = layer_member_seed(base_seed, member_idx, layer_idx)
    g = torch.Generator(device="cpu").manual_seed(seed)
    total = out_features * rank + in_features * rank
    flat = torch.randn(total, generator=g, dtype=dtype)
    A = flat[:out_features * rank].reshape(out_features, rank).to(device)
    B = flat[out_features * rank:].reshape(in_features, rank).to(device)
    return A, B


def draw_layer_factors_batch(base_seed: int, n_pop: int, layer_idx: int,
                             out_features: int, in_features: int, rank: int,
                             device, dtype=torch.float32):
    """Generate (A, B) for ALL members at one layer. Used in pass 2.

    A: (N, out, rank), B: (N, in, rank).
    """
    A = torch.empty(n_pop, out_features, rank, dtype=dtype)
    B = torch.empty(n_pop, in_features, rank, dtype=dtype)
    for i in range(n_pop):
        A[i], B[i] = draw_layer_factors(base_seed, i, layer_idx,
                                         out_features, in_features, rank,
                                         device="cpu", dtype=dtype)
    return A.to(device), B.to(device)
