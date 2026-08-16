"""Forward-hook EGGROLL perturbation for HuggingFace models.

Each nn.Linear in the model gets an EggrollHook that adds the low-rank
perturbation during the forward pass:

    output += sigma/sqrt(r) * (input @ B_i) @ A_i.T

Factors (A_i, B_i) are generated from the member's seed on-the-fly and
discarded immediately. Memory cost is O(1 layer) per hook, not O(all layers).

Usage:
    manager = HookManager(model, sigma=0.001, rank=1)
    manager.activate(gen_seed=42, member_idx=3)   # turn on perturbation
    output = model.generate(...)                    # perturbed forward pass
    manager.deactivate()                            # back to base model
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .noise import draw_layer_factors


class EggrollHook:
    """Forward hook for one nn.Linear implementing EGGROLL perturbation."""

    def __init__(self, layer_idx: int, out_features: int, in_features: int,
                 rank: int, sigma: float, device, dtype):
        self.layer_idx = layer_idx
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.sigma = sigma
        self.device = device
        self.dtype = dtype
        self._active = False
        self._base_seed = 0
        self._member_idx = 0

    def activate(self, base_seed: int, member_idx: int):
        self._active = True
        self._base_seed = base_seed
        self._member_idx = member_idx

    def deactivate(self):
        self._active = False

    def __call__(self, module: nn.Module, args, output):
        if not self._active:
            return output

        x = args[0]  # (batch, seq_len, in_features) or (batch, in_features)
        A, B = draw_layer_factors(
            self._base_seed, self._member_idx, self.layer_idx,
            self.out_features, self.in_features, self.rank,
            device=x.device, dtype=x.dtype,
        )
        scale = self.sigma / (self.rank ** 0.5)
        # x @ B: (..., rank), then @ A.T: (..., out_features)
        return output + scale * (x @ B) @ A.T


class HookManager:
    """Manages EGGROLL perturbation hooks across an entire model.

    Discovers all nn.Linear layers, assigns each a layer index, and
    registers forward hooks. The hooks are inactive by default; call
    activate(seed, member_idx) before each member's forward pass.
    """

    def __init__(self, model: nn.Module, sigma: float, rank: int = 1,
                 perturbable_filter=None):
        """
        Args:
            model: HuggingFace model
            sigma: perturbation scale
            rank: factor rank (1 is the EGGROLL default)
            perturbable_filter: optional callable(name, module) -> bool
                to restrict which layers get perturbed. Default: all nn.Linear.
        """
        self.model = model
        self.sigma = sigma
        self.rank = rank
        self.hooks: list[EggrollHook] = []
        self.handles: list[torch.utils.hooks.RemovableHook] = []
        self.layer_info: list[tuple[str, int, int]] = []  # (name, out, in)

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        layer_idx = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if perturbable_filter and not perturbable_filter(name, module):
                continue

            out_feat = module.out_features
            in_feat = module.in_features
            self.layer_info.append((name, out_feat, in_feat))

            hook = EggrollHook(
                layer_idx=layer_idx,
                out_features=out_feat,
                in_features=in_feat,
                rank=rank,
                sigma=sigma,
                device=device,
                dtype=dtype,
            )
            # Use register_forward_hook with the 3-arg signature
            handle = module.register_forward_hook(hook)
            self.hooks.append(hook)
            self.handles.append(handle)
            layer_idx += 1

    @property
    def n_layers(self) -> int:
        return len(self.hooks)

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        """name -> (out_features, in_features) for all perturbable layers."""
        return {name: (out, inp) for name, out, inp in self.layer_info}

    @property
    def n_params(self) -> int:
        return sum(o * i for _, o, i in self.layer_info)

    def set_sigma(self, sigma: float):
        self.sigma = sigma
        for hook in self.hooks:
            hook.sigma = sigma

    def activate(self, gen_seed: int, member_idx: int):
        """Activate all hooks for a specific member."""
        for hook in self.hooks:
            hook.activate(gen_seed, member_idx)

    def deactivate(self):
        """Deactivate all hooks (back to unperturbed model)."""
        for hook in self.hooks:
            hook.deactivate()

    def remove(self):
        """Permanently remove all hooks from the model."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hooks.clear()

    # ---- Partition support ----

    def auto_partitions(self, n_groups: int | None = None) -> dict[str, list[int]]:
        """Generate layer-group partitions.

        Groups consecutive transformer layers together. Each partition is a
        list of layer indices (into self.layer_info).

        Args:
            n_groups: number of partitions. None = auto (ceil(n_layers_model / 4)
                      where n_layers_model is the number of transformer layers,
                      not the number of nn.Linear modules).
        """
        # Detect transformer layers from names like "model.layers.0.self_attn.q_proj"
        layer_numbers = set()
        for name, _, _ in self.layer_info:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_numbers.add(int(parts[i + 1]))

        n_transformer_layers = max(layer_numbers) + 1 if layer_numbers else 1

        if n_groups is None:
            n_groups = max(1, -(-n_transformer_layers // 4))  # ceil div

        # Group by transformer layer number
        partitions: dict[str, list[int]] = {}
        layers_per_group = max(1, -(-n_transformer_layers // n_groups))

        for idx, (name, _, _) in enumerate(self.layer_info):
            # Extract transformer layer number
            t_layer = None
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    t_layer = int(parts[i + 1])
            if t_layer is None:
                # Embedding/head layers go in group "other"
                group_name = "other"
            else:
                group_idx = min(t_layer // layers_per_group, n_groups - 1)
                group_name = f"block_{group_idx}"

            partitions.setdefault(group_name, []).append(idx)

        return partitions
