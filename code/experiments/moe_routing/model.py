"""Toy Mixture-of-Experts model for the ES routing experiment.

Architecture:
    input -> embedding -> router -> top-1 expert -> output head

    - embedding:  (vocab_size, hidden_dim) linear projection
    - router:     (hidden_dim, K) linear layer, argmax selects one expert
    - experts:    K independent 2-layer MLPs  (hidden -> 4*hidden -> hidden)
    - output:     (hidden_dim, n_classes) linear head

The model does sequence classification on a synthetic modular arithmetic task:
given a sequence of digits and an operator, predict the result mod n_classes.

~530K parameters at hidden=128, K=4; ~2.1M at hidden=256.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Two-layer MLP expert: hidden -> 4*hidden -> hidden with GELU."""

    def __init__(self, hidden: int):
        super().__init__()
        self.up = nn.Linear(hidden, 4 * hidden)
        self.down = nn.Linear(4 * hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class ToyMoE(nn.Module):
    """Minimal MoE: shared embedding + K expert MLPs + router + output head.

    The router is a simple linear layer (hidden -> K). During forward, we take
    argmax to get hard routing (the discrete decision ES should handle well).

    For gradient-based training of the router, we use softmax + weighted sum
    (soft routing) so gradients flow. The ``hard`` flag controls this.
    """

    def __init__(self, vocab_size: int = 32, seq_len: int = 8,
                 hidden: int = 256, n_experts: int = 4, n_classes: int = 16,
                 seed: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_experts = n_experts
        self.n_classes = n_classes

        torch.manual_seed(seed)
        self.embedding = nn.Linear(vocab_size, hidden)
        self.experts = nn.ModuleList([ExpertMLP(hidden) for _ in range(n_experts)])
        self.router = nn.Linear(hidden, n_experts, bias=False)
        self.output_head = nn.Linear(hidden, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Router: small init so early routing is roughly uniform.
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x: torch.Tensor, hard: bool = True,
                return_routing: bool = False):
        """
        Args:
            x: (batch, seq_len) long tensor of token ids
            hard: if True, argmax routing (discrete); if False, softmax weighting
            return_routing: if True, also return the routing logits

        Returns:
            logits: (batch, n_classes)
            routing_logits: (batch, n_experts) -- only if return_routing=True
        """
        # Embed and pool over sequence
        one_hot = F.one_hot(x.long(), self.vocab_size).float()  # (B, S, V)
        h = self.embedding(one_hot)                              # (B, S, H)
        h = h.mean(dim=1)                                       # (B, H) -- mean pool

        # Routing
        routing_logits = self.router(h)                          # (B, K)

        if hard:
            # Hard routing: argmax, run only the selected expert per sample
            chosen = routing_logits.argmax(dim=-1)               # (B,)
            out = torch.zeros_like(h)
            for k in range(self.n_experts):
                mask = chosen == k
                if mask.any():
                    out[mask] = self.experts[k](h[mask])
        else:
            # Soft routing: weighted sum of all experts (for gradient training)
            weights = F.softmax(routing_logits, dim=-1)          # (B, K)
            out = torch.zeros_like(h)
            for k in range(self.n_experts):
                out = out + weights[:, k:k+1] * self.experts[k](h)

        logits = self.output_head(out + h)                       # residual + head

        if return_routing:
            return logits, routing_logits
        return logits

    def freeze_experts(self):
        """Freeze everything except the router."""
        for p in self.embedding.parameters():
            p.requires_grad = False
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = False
        for p in self.output_head.parameters():
            p.requires_grad = False
        # Router stays trainable
        for p in self.router.parameters():
            p.requires_grad = True

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def router_params(self) -> int:
        return sum(p.numel() for p in self.router.parameters())


# -------------------------------------------------------------------- dataset

def make_modular_arithmetic_data(n_samples: int, seq_len: int = 8,
                                  vocab_size: int = 32, n_classes: int = 16,
                                  seed: int = 0):
    """Synthetic task: sum of token values mod n_classes.

    Each token is an integer in [0, vocab_size). The label is
    sum(tokens) mod n_classes. Simple enough that experts can learn it,
    interesting enough that different experts can specialise on different
    input ranges.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
    y = x.sum(dim=1) % n_classes
    return x, y


def make_split_data(n_train: int = 8000, n_val: int = 2000, **kwargs):
    """Train/val split with different seeds."""
    x_tr, y_tr = make_modular_arithmetic_data(n_train, seed=kwargs.get('seed', 0),
                                               **{k: v for k, v in kwargs.items()
                                                  if k != 'seed'})
    x_va, y_va = make_modular_arithmetic_data(n_val, seed=kwargs.get('seed', 0) + 99999,
                                               **{k: v for k, v in kwargs.items()
                                                  if k != 'seed'})
    return x_tr, y_tr, x_va, y_va
