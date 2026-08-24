"""Experimental operators for CPU benchmark — Directions A, B, D.

A: GuidedSampler — bias perturbations toward recent gradient directions (SGES-inspired)
B: ArchiveSampler — seed population from archive of high-fitness perturbations
D: FinePartitionedSampler — partition within parameter tensors for P > n_params
"""

from __future__ import annotations

import torch

from evolve.core.noise import draw_factors, chunk_seed


# ---------------------------------------------------------------- Direction A

class GuidedSampler:
    """Blend recent gradient directions into perturbation factors.

    After each generation the trainer feeds back the gradient direction
    dW = sum_i w_i A_i B_i^T. We store its rank-1 SVD (u, v) and, in
    subsequent generations, additively bias every member's factors toward
    one of the stored directions.

    SGES (Liu et al., IJCAI 2020) proved this reduces gradient estimator
    variance with bounded bias. The key insight: don't throw away
    inter-generation information. The directions the optimizer recently
    moved in are informative about where to explore next.

    Determinism: the blend depends only on stored directions (which don't
    change within a generation) and the member's seeded factors, so pass 1
    and pass 2 produce identical results.
    """

    def __init__(self, k: int = 8, alpha: float = 0.5):
        self.k = k              # number of directions to store
        self.alpha = alpha      # blend strength (0 = pure isotropic, 1 = pure guided)
        self._directions: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def record_gradient(self, delta: dict[str, torch.Tensor]):
        """Store the dominant direction of the latest gradient estimate."""
        for name, dW in delta.items():
            m, n = dW.shape
            if min(m, n) < 1:
                continue
            U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
            # Dominant rank-1 component, scaled to unit-variance entries
            u = U[:, 0:1] * (m ** 0.5)     # (m, 1), entries ~ O(1)
            v = Vt[0:1, :].T * (n ** 0.5)  # (n, 1), entries ~ O(1)
            if name not in self._directions:
                self._directions[name] = []
            self._directions[name].append((u.detach(), v.detach()))
            if len(self._directions[name]) > self.k:
                self._directions[name].pop(0)

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0):
        fac = draw_factors(seed, shapes, n_pop, rank, device,
                           index_offset=index_offset)

        if not self._directions:
            return fac

        for name in fac:
            dirs = self._directions.get(name)
            if not dirs:
                continue

            A, B = fac[name]  # (N, m, r), (N, n, r)
            n_dirs = len(dirs)

            for i in range(A.shape[0]):
                global_idx = index_offset + i
                j = global_idx % n_dirs
                u, v = dirs[j]  # (m, 1), (n, 1)
                # Blend: factor = (1-α) * random + α * direction
                A[i] = (1 - self.alpha) * A[i] + self.alpha * u.to(device)
                B[i] = (1 - self.alpha) * B[i] + self.alpha * v.to(device)

            fac[name] = (A, B)

        return fac

    def importance(self, factors, n_pop):
        return None  # biased toward gradient — intentional, not correctable


# ---------------------------------------------------------------- Direction B

class ArchiveSampler:
    """Seed a fraction of the population from archived high-fitness perturbations.

    After each generation the trainer identifies top-performing members and
    stores their factor-pairs. In subsequent generations, every K-th member
    (by global index) draws its factors from the archive + noise instead of
    from scratch.

    This gives temporal persistence to discovered diverse directions without
    maintaining a persistent population. It's lightweight MAP-Elites adapted
    to EGGROLL's factored format.

    Memory: at rank=1, each archived member is ~500 floats for the 94k-param
    TinyPolicy. An archive of 64 entries = 32k floats = 128 KB. Negligible.
    """

    def __init__(self, archive_size: int = 64, seed_fraction: float = 0.25,
                 noise_scale: float = 0.3):
        self.archive_size = archive_size
        self.seed_fraction = seed_fraction
        self.noise_scale = noise_scale
        self._archive: list[tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]] = []

    def update_archive(self, factors: dict, fitness: torch.Tensor,
                       top_k: int = 8):
        """Add top-k members from the current generation to the archive."""
        n = min(top_k, fitness.shape[0])
        top_idx = fitness.argsort(descending=True)[:n]

        for idx in top_idx:
            entry = {}
            for name, (A, B) in factors.items():
                entry[name] = (A[idx].detach().clone(), B[idx].detach().clone())
            fit = float(fitness[idx])
            self._archive.append((entry, fit))

        # Trim, keeping highest fitness
        if len(self._archive) > self.archive_size:
            self._archive.sort(key=lambda x: x[1], reverse=True)
            self._archive = self._archive[:self.archive_size]

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0):
        fac = draw_factors(seed, shapes, n_pop, rank, device,
                           index_offset=index_offset)

        if not self._archive:
            return fac

        # Every K-th member by global index is seeded from the archive
        K = max(1, round(1.0 / self.seed_fraction))

        for i in range(n_pop):
            global_idx = index_offset + i
            if global_idx % K != 0:
                continue

            arch_idx = (global_idx // K) % len(self._archive)
            arch_factors, _ = self._archive[arch_idx]

            for name in fac:
                if name not in arch_factors:
                    continue
                A, B = fac[name]
                a_arch, b_arch = arch_factors[name]
                # Archived direction + scaled noise for exploration
                A[i] = a_arch.to(device) + self.noise_scale * A[i]
                B[i] = b_arch.to(device) + self.noise_scale * B[i]

        return fac

    def importance(self, factors, n_pop):
        return None


# ---------------------------------------------------------------- Direction D

class FinePartitionedSampler:
    """Partitioned perturbation at sub-tensor granularity.

    The regular PartitionedSampler assigns one whole parameter matrix per part.
    With only 3 matrices (emb, h, out) in TinyPolicy, P is capped at 3.

    This sampler splits large matrices into row-blocks so P can be 8, 16, or
    more. Each member still perturbs only n_active parts, but the parts are
    now sub-blocks of the parameter tensors.

    The auto_partition logic allocates sub-blocks proportionally to parameter
    count: the 24k-param hidden layer gets most of the parts, while the
    224-param embedding gets one.
    """

    def __init__(self, n_groups: int = 8, n_active: int = 1):
        self.n_groups = n_groups
        self.n_active = n_active
        self._parts: dict[str, list[tuple[str, int, int]]] | None = None
        self._last_mask: torch.Tensor | None = None
        self._last_pi: torch.Tensor | None = None
        self._part_names: list[str] = []

    def _auto_partition(self, shapes: dict[str, tuple[int, int]]):
        """Split parameters into n_groups parts proportional to size."""
        total_params = sum(m * n for m, n in shapes.values())
        parts = {}
        budget = self.n_groups

        # Sort by size descending — allocate more sub-blocks to bigger matrices
        items = sorted(shapes.items(), key=lambda x: x[1][0] * x[1][1],
                       reverse=True)

        allocated = 0
        for idx, (name, (m, n)) in enumerate(items):
            remaining_items = len(items) - idx
            remaining_budget = budget - allocated

            # How many sub-blocks for this param?
            param_frac = (m * n) / total_params
            n_sub = max(1, round(remaining_budget * param_frac))
            # Ensure we leave at least 1 per remaining item
            n_sub = min(n_sub, remaining_budget - (remaining_items - 1))
            n_sub = min(n_sub, m)  # can't have more blocks than rows
            n_sub = max(1, n_sub)

            block_size = m // n_sub
            for j in range(n_sub):
                r_start = j * block_size
                r_end = m if j == n_sub - 1 else (j + 1) * block_size
                part_name = f"{name}_{j}" if n_sub > 1 else name
                parts[part_name] = [(name, r_start, r_end)]

            allocated += n_sub

        return parts

    def draw(self, seed, shapes, n_pop, rank, device, index_offset=0):
        if self._parts is None:
            self._parts = self._auto_partition(shapes)

        part_names = list(self._parts)
        self._part_names = part_names
        n_parts = len(part_names)
        pi = torch.full((n_parts,), 1.0 / n_parts)

        # Generate mask: each member selects n_active parts
        mask = torch.zeros(n_pop, n_parts)
        for j in range(n_pop):
            g = torch.Generator(device="cpu").manual_seed(
                (int(seed) ^ 0x5EED) + 0x9E37 * (index_offset + j))
            pick = torch.multinomial(pi, min(self.n_active, n_parts),
                                     replacement=False, generator=g)
            mask[j, pick] = 1.0
        self._last_mask = mask.to(device)
        self._last_pi = pi.to(device)

        # Draw factors normally
        fac = draw_factors(seed, shapes, n_pop, rank, device,
                           index_offset=index_offset)

        # Zero out A rows for unselected sub-blocks
        # First, save originals and zero everything
        original = {}
        for name in fac:
            A, B = fac[name]
            original[name] = A.clone()
            A.zero_()

        # Restore selected sub-blocks
        for p_idx, part_name in enumerate(part_names):
            keep = mask[:, p_idx].to(device).view(-1, 1, 1)  # (N, 1, 1)
            for param_name, r_start, r_end in self._parts[part_name]:
                A, _ = fac[param_name]
                A[:, r_start:r_end, :] += (
                    original[param_name][:, r_start:r_end, :] * keep
                )

        return fac

    def importance(self, factors, n_pop):
        """Importance correction: 1/(pi_active * P)."""
        if self._last_mask is None:
            return None
        pi_eff = (self._last_mask * self._last_pi).sum(dim=1).clamp_min(1e-8)
        return 1.0 / (pi_eff * len(self._part_names))

    @property
    def part_names(self):
        return self._part_names
