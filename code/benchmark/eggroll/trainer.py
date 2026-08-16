"""ES training loop for HuggingFace models with hook-based perturbation.

Two-pass structure per generation:
    pass 1   for each member: activate hooks -> generate -> score -> collect fitness
    ----     compute weights over the WHOLE population
    pass 2   for each layer: regenerate all members' factors -> accumulate dW -> apply

Pass 2 regenerates factors by layer (not by member) to vectorise the accumulation
across the population. Memory: O(N * max_dim * rank) per layer, which is trivial.

CRN strategy: greedy decoding during training. All members see the same problem
batch (CRN for problem selection). Greedy decoding is deterministic given the same
input, so member differences are purely from weight perturbation. Temperature
sampling is used only for pass@k evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from .hooks import HookManager
from .noise import draw_layer_factors_batch

# Import proven math from the existing framework
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolve.core.noise import chunk_seed, member_seed


def rank_normalise(F: torch.Tensor) -> torch.Tensor:
    """Centred rank transform with tied ranks averaged.

    Identical to evolve.es.eggroll.rank_normalise — duplicated here so the
    benchmark is self-contained for the friend's checkout.
    """
    N = F.numel()
    order = F.argsort()
    ranks = torch.empty_like(F)
    ranks[order] = torch.arange(N, dtype=F.dtype, device=F.device)
    sortedF = F[order]
    uniq, inv, counts = torch.unique(sortedF, return_inverse=True, return_counts=True)
    sums = torch.zeros(uniq.numel(), dtype=F.dtype, device=F.device)
    sums.index_add_(0, inv, torch.arange(N, dtype=F.dtype, device=F.device))
    ranks[order] = (sums / counts)[inv]
    return ranks / max(N - 1, 1) - 0.5


@dataclass
class ESConfig:
    n_pop: int = 32
    rank: int = 1
    sigma: float = 0.001
    alpha: float = 5e-4
    generations: int = 200
    seed: int = 0


@dataclass
class GenRecord:
    generation: int
    fitness_mean: float
    fitness_max: float
    fitness_std: float
    sigma: float
    seconds: float
    extra: dict = field(default_factory=dict)


class ESTrainer:
    """EGGROLL ES on a HuggingFace model via forward hooks.

    Args:
        model: HuggingFace causal LM (already loaded)
        tokenizer: corresponding tokenizer
        hook_manager: HookManager wrapping the model
        evaluate_fn: callable(model, tokenizer, gen_seed, greedy) -> (fitness_N, per_example_N_M)
            Takes the model (with hooks active for one member at a time),
            and returns per-member scalar fitness and per-example rewards.
        config: ESConfig
        sigma_rule: optional sigma adaptation rule (e.g. ResolutionRule)
        parent_eval_fn: callable(model, tokenizer, gen_seed) -> float
            Unperturbed parent fitness under the same CRN seed. Required for sigma rules.
    """

    def __init__(self, model: nn.Module, tokenizer, hook_manager: HookManager,
                 evaluate_fn: Callable, config: ESConfig,
                 sigma_rule=None, parent_eval_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = hook_manager
        self.evaluate_fn = evaluate_fn
        self.cfg = config
        self.sigma_rule = sigma_rule
        self.parent_eval_fn = parent_eval_fn
        self.history: list[GenRecord] = []
        self._sigma = config.sigma

    def step(self, generation: int) -> GenRecord:
        t0 = time.time()
        cfg = self.cfg
        gen_seed = chunk_seed(cfg.seed * 1_000_003, generation)

        # Update hook sigma
        self.hooks.set_sigma(self._sigma)

        # ---- pass 1: evaluate each member ------------------------------------
        all_fitness = []
        all_per_example = []

        for member_idx in range(cfg.n_pop):
            self.hooks.activate(gen_seed, member_idx)
            f_scalar, f_per_example = self.evaluate_fn(
                self.model, self.tokenizer, gen_seed, greedy=True)
            all_fitness.append(f_scalar)
            all_per_example.append(f_per_example)

        self.hooks.deactivate()

        fitness = torch.tensor(all_fitness, dtype=torch.float32)
        per_example = torch.stack(all_per_example) if all_per_example[0] is not None \
            else torch.zeros(cfg.n_pop, 1)

        # ---- sigma adaptation (paired parent) --------------------------------
        if self.sigma_rule is not None and self.parent_eval_fn is not None:
            parent_f = self.parent_eval_fn(self.model, self.tokenizer, gen_seed)
            self._sigma = self.sigma_rule.update(fitness, parent_f)
            self.hooks.set_sigma(self._sigma)

        # ---- weights over the WHOLE population --------------------------------
        weights = rank_normalise(fitness)

        # ---- pass 2: accumulate update layer by layer -------------------------
        scale = cfg.alpha / (cfg.n_pop * self._sigma)
        r_scale = cfg.rank ** -0.5

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        for layer_idx, (layer_name, out_feat, in_feat) in enumerate(
                self.hooks.layer_info):
            # Generate ALL members' factors for this layer
            A, B = draw_layer_factors_batch(
                gen_seed, cfg.n_pop, layer_idx,
                out_feat, in_feat, cfg.rank,
                device=device, dtype=dtype,
            )
            # dW = sum_i w_i * (1/sqrt(r)) * A_i @ B_i^T
            # Using einsum: "i,imr,inr->mn"
            dW = torch.einsum("i,imr,inr->mn",
                              weights.to(dtype).to(device), A, B) * r_scale

            # Apply to model parameter
            param = self._get_param(layer_name)
            param.data += scale * dW

        rec = GenRecord(
            generation=generation,
            fitness_mean=float(fitness.mean()),
            fitness_max=float(fitness.max()),
            fitness_std=float(fitness.std()),
            sigma=self._sigma,
            seconds=time.time() - t0,
        )
        self.history.append(rec)
        return rec

    def _get_param(self, layer_name: str) -> nn.Parameter:
        """Navigate the model's named_modules to find the parameter."""
        parts = layer_name.split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module.weight

    def run(self, generations: int | None = None, callback=None):
        n = generations if generations is not None else self.cfg.generations
        for gen in range(n):
            rec = self.step(gen)
            if callback is not None:
                callback(rec)
        return self.history


class PartitionedESTrainer(ESTrainer):
    """ES with partitioned perturbation: each member perturbs one layer-group.

    The partition mask is drawn deterministically per member (keyed on gen_seed
    and member_idx), so chunking and ordering don't matter.
    """

    def __init__(self, *args, partitions: dict[str, list[int]] | None = None,
                 n_active: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.partitions = partitions or self.hooks.auto_partitions()
        self.n_active = n_active
        self._part_names = list(self.partitions.keys())
        self._last_assignments: dict[int, list[str]] = {}  # member -> active parts

    def _draw_mask(self, gen_seed: int, member_idx: int) -> list[str]:
        """Which parts this member perturbs. Deterministic."""
        g = torch.Generator(device="cpu").manual_seed(
            (gen_seed ^ 0x5EED) + 0x9E37 * member_idx)
        n_parts = len(self._part_names)
        probs = torch.full((n_parts,), 1.0 / n_parts)
        pick = torch.multinomial(probs, min(self.n_active, n_parts),
                                 replacement=False, generator=g)
        return [self._part_names[i] for i in pick]

    def _active_layers(self, active_parts: list[str]) -> set[int]:
        """Layer indices that belong to any active part."""
        layers = set()
        for part in active_parts:
            layers.update(self.partitions[part])
        return layers

    def step(self, generation: int) -> GenRecord:
        t0 = time.time()
        cfg = self.cfg
        gen_seed = chunk_seed(cfg.seed * 1_000_003, generation)
        self.hooks.set_sigma(self._sigma)

        # ---- pass 1: evaluate with masked perturbation -----------------------
        all_fitness = []
        all_per_example = []
        self._last_assignments = {}

        for member_idx in range(cfg.n_pop):
            active_parts = self._draw_mask(gen_seed, member_idx)
            self._last_assignments[member_idx] = active_parts
            active_layers = self._active_layers(active_parts)

            # Only activate hooks for layers in active parts
            for hook_idx, hook in enumerate(self.hooks.hooks):
                if hook_idx in active_layers:
                    hook.activate(gen_seed, member_idx)
                else:
                    hook.deactivate()

            f_scalar, f_per_example = self.evaluate_fn(
                self.model, self.tokenizer, gen_seed, greedy=True)
            all_fitness.append(f_scalar)
            all_per_example.append(f_per_example)

        self.hooks.deactivate()

        fitness = torch.tensor(all_fitness, dtype=torch.float32)
        per_example = torch.stack(all_per_example) if all_per_example[0] is not None \
            else torch.zeros(cfg.n_pop, 1)

        # ---- sigma adaptation ------------------------------------------------
        if self.sigma_rule is not None and self.parent_eval_fn is not None:
            parent_f = self.parent_eval_fn(self.model, self.tokenizer, gen_seed)
            self._sigma = self.sigma_rule.update(fitness, parent_f)
            self.hooks.set_sigma(self._sigma)

        # ---- weights ---------------------------------------------------------
        weights = rank_normalise(fitness)

        # ---- per-part utility ------------------------------------------------
        part_utilities = {}
        for part_name in self._part_names:
            member_indices = [i for i, parts in self._last_assignments.items()
                              if part_name in parts]
            if member_indices:
                w_part = weights[member_indices]
                part_utilities[part_name] = float(w_part.abs().mean())
            else:
                part_utilities[part_name] = 0.0

        # ---- importance correction ------------------------------------------
        # Each member perturbed n_active of P parts with uniform probability.
        # Effective probability per member = n_active / P.
        # Importance weight = P / n_active (constant for uniform sampling).
        n_parts = len(self._part_names)
        importance = n_parts / self.n_active

        # ---- pass 2: accumulate update with masking --------------------------
        scale = cfg.alpha / (cfg.n_pop * self._sigma) * importance
        r_scale = cfg.rank ** -0.5
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        for layer_idx, (layer_name, out_feat, in_feat) in enumerate(
                self.hooks.layer_info):
            # Find which members perturbed this layer
            layer_members = []
            for member_idx in range(cfg.n_pop):
                active_layers = self._active_layers(
                    self._last_assignments[member_idx])
                if layer_idx in active_layers:
                    layer_members.append(member_idx)

            if not layer_members:
                continue

            # Generate factors only for active members
            A = torch.empty(len(layer_members), out_feat, cfg.rank, dtype=dtype)
            B = torch.empty(len(layer_members), in_feat, cfg.rank, dtype=dtype)
            w_active = torch.empty(len(layer_members), dtype=dtype)

            for j, mi in enumerate(layer_members):
                from .noise import draw_layer_factors
                A[j], B[j] = draw_layer_factors(
                    gen_seed, mi, layer_idx, out_feat, in_feat, cfg.rank,
                    device="cpu", dtype=dtype)
                w_active[j] = weights[mi]

            A, B, w_active = A.to(device), B.to(device), w_active.to(device)
            dW = torch.einsum("i,imr,inr->mn", w_active, A, B) * r_scale

            param = self._get_param(layer_name)
            param.data += scale * dW

        rec = GenRecord(
            generation=generation,
            fitness_mean=float(fitness.mean()),
            fitness_max=float(fitness.max()),
            fitness_std=float(fitness.std()),
            sigma=self._sigma,
            seconds=time.time() - t0,
            extra={"part_utilities": part_utilities},
        )
        self.history.append(rec)
        return rec


class SelectiveESTrainer(PartitionedESTrainer):
    """Partitioned ES with utility-gated accumulation.

    Same as PartitionedESTrainer, but in pass 2, only applies updates to
    parts with above-median utility. Parts not useful for the target task
    stay frozen -> sparse updates -> less forgetting.

    This is the DiPEC-inspired idea: NOT (2+1), NOT population subsampling.
    Proper population (N=32+), partitioned sampling, selection on *which
    parameters to update*.
    """

    def __init__(self, *args, threshold: str = "median", **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def step(self, generation: int) -> GenRecord:
        t0 = time.time()
        cfg = self.cfg
        gen_seed = chunk_seed(cfg.seed * 1_000_003, generation)
        self.hooks.set_sigma(self._sigma)

        # ---- pass 1: evaluate (same as PartitionedESTrainer) -----------------
        all_fitness = []
        all_per_example = []
        self._last_assignments = {}

        for member_idx in range(cfg.n_pop):
            active_parts = self._draw_mask(gen_seed, member_idx)
            self._last_assignments[member_idx] = active_parts
            active_layers = self._active_layers(active_parts)

            for hook_idx, hook in enumerate(self.hooks.hooks):
                if hook_idx in active_layers:
                    hook.activate(gen_seed, member_idx)
                else:
                    hook.deactivate()

            f_scalar, f_per_example = self.evaluate_fn(
                self.model, self.tokenizer, gen_seed, greedy=True)
            all_fitness.append(f_scalar)
            all_per_example.append(f_per_example)

        self.hooks.deactivate()

        fitness = torch.tensor(all_fitness, dtype=torch.float32)
        per_example = torch.stack(all_per_example) if all_per_example[0] is not None \
            else torch.zeros(cfg.n_pop, 1)

        # ---- sigma adaptation ------------------------------------------------
        if self.sigma_rule is not None and self.parent_eval_fn is not None:
            parent_f = self.parent_eval_fn(self.model, self.tokenizer, gen_seed)
            self._sigma = self.sigma_rule.update(fitness, parent_f)
            self.hooks.set_sigma(self._sigma)

        # ---- weights ---------------------------------------------------------
        weights = rank_normalise(fitness)

        # ---- per-part utility ------------------------------------------------
        part_utilities = {}
        for part_name in self._part_names:
            member_indices = [i for i, parts in self._last_assignments.items()
                              if part_name in parts]
            if member_indices:
                w_part = weights[member_indices]
                part_utilities[part_name] = float(w_part.abs().mean())
            else:
                part_utilities[part_name] = 0.0

        # ---- selective gating: only update high-utility parts ----------------
        import statistics
        vals = list(part_utilities.values())
        if self.threshold == "median" and vals:
            thresh = statistics.median(vals)
        else:
            thresh = float(self.threshold) if self.threshold != "median" else 0.0

        active_update_parts = {p for p, u in part_utilities.items() if u >= thresh}
        active_update_layers = set()
        for part in active_update_parts:
            active_update_layers.update(self.partitions[part])

        # ---- importance correction ------------------------------------------
        n_parts = len(self._part_names)
        importance = n_parts / self.n_active

        # ---- pass 2: accumulate update (only for active parts) ---------------
        scale = cfg.alpha / (cfg.n_pop * self._sigma) * importance
        r_scale = cfg.rank ** -0.5
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        for layer_idx, (layer_name, out_feat, in_feat) in enumerate(
                self.hooks.layer_info):
            # Skip if this layer's part was gated out
            if layer_idx not in active_update_layers:
                continue

            # Find which members perturbed this layer
            layer_members = []
            for member_idx in range(cfg.n_pop):
                active_layers = self._active_layers(
                    self._last_assignments[member_idx])
                if layer_idx in active_layers:
                    layer_members.append(member_idx)

            if not layer_members:
                continue

            A = torch.empty(len(layer_members), out_feat, cfg.rank, dtype=dtype)
            B = torch.empty(len(layer_members), in_feat, cfg.rank, dtype=dtype)
            w_active = torch.empty(len(layer_members), dtype=dtype)

            for j, mi in enumerate(layer_members):
                from .noise import draw_layer_factors
                A[j], B[j] = draw_layer_factors(
                    gen_seed, mi, layer_idx, out_feat, in_feat, cfg.rank,
                    device="cpu", dtype=dtype)
                w_active[j] = weights[mi]

            A, B, w_active = A.to(device), B.to(device), w_active.to(device)
            dW = torch.einsum("i,imr,inr->mn", w_active, A, B) * r_scale

            param = self._get_param(layer_name)
            param.data += scale * dW

        rec = GenRecord(
            generation=generation,
            fitness_mean=float(fitness.mean()),
            fitness_max=float(fitness.max()),
            fitness_std=float(fitness.std()),
            sigma=self._sigma,
            seconds=time.time() - t0,
            extra={
                "part_utilities": part_utilities,
                "active_parts": list(active_update_parts),
                "n_parts_updated": len(active_update_parts),
                "n_parts_total": n_parts,
            },
        )
        self.history.append(rec)
        return rec
