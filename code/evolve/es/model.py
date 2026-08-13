"""A small char-level LM with EGGROLL-style population forward pass.

The point of this file is the factorised forward:

    y_i = x W^T + sigma * (x B_i) A_i^T

so the shared matmul is computed once for the whole population and the per-member
work is low-rank. That is the whole EGGROLL trick, and it also hands us the
per-member activation delta `(x B_i) A_i^T` for free -- which Proposal A uses as a
functional signature at zero marginal cost.
"""

import torch
import torch.nn.functional as F


class TinyLM:
    """embedding -> concat context -> hidden -> logits. Three perturbable matrices."""

    def __init__(self, vocab, ctx, emb=16, hidden=64, device="cpu", seed=0):
        g = torch.Generator().manual_seed(seed)
        self.vocab, self.ctx, self.emb, self.hidden = vocab, ctx, emb, hidden
        self.device = device
        self.W = {
            "emb": torch.randn(vocab, emb, generator=g) * 0.3,
            "h": torch.randn(hidden, ctx * emb, generator=g) * (1.0 / (ctx * emb) ** 0.5),
            "out": torch.randn(vocab, hidden, generator=g) * (1.0 / hidden**0.5),
        }
        self.W = {k: v.to(device) for k, v in self.W.items()}

    @property
    def shapes(self):
        return {k: tuple(v.shape) for k, v in self.W.items()}

    def forward_population(self, X, fac, sigma, rank, capture_signature=True):
        """X: (B, ctx) int64. fac: {name: (A, B)} with A (N,m,r), B (N,n,r).

        Returns logits (N, B, vocab) and optionally the raw activation delta at the
        output layer, (N, B, vocab), which is Proposal A's Tier-1 signature.
        """
        N = fac["h"][0].shape[0]
        s = sigma / (rank**0.5)

        # --- embedding: row lookup of (W + s A B^T) ---
        e0 = self.W["emb"][X]                                   # (B, ctx, emb)
        A, Bf = fac["emb"]                                      # (N,vocab,r), (N,emb,r)
        a_rows = A[:, X, :]                                     # (N, B, ctx, r)
        e = e0.unsqueeze(0) + s * torch.einsum("nbcr,ndr->nbcd", a_rows, Bf)
        h_in = e.flatten(start_dim=2)                           # (N, B, ctx*emb)

        # --- hidden: y = x W^T + s (x B) A^T ---
        A, Bf = fac["h"]
        base = h_in @ self.W["h"].T                             # (N, B, hidden)
        h = torch.tanh(base + s * torch.einsum("nbi,nir,nhr->nbh", h_in, Bf, A))

        # --- output ---
        A, Bf = fac["out"]
        base = h @ self.W["out"].T
        delta = s * torch.einsum("nbi,nir,nvr->nbv", h, Bf, A)
        return base + delta, (delta if capture_signature else None)

    def total_output_delta(self, X, logits):
        """logits_i - logits_base: the member's *total* functional change.

        Note the difference from the `delta` returned above, which is only the
        output layer's own low-rank contribution and therefore misses the effect of
        perturbing earlier layers. In EGGROLL the base forward is computed anyway
        (it is the shared matmul), so this is nearly free too -- but it is NOT the
        same object, and E0 shows the distinction matters a great deal."""
        return logits - self.forward_base(X).unsqueeze(0)

    def forward_base(self, X):
        e = self.W["emb"][X].flatten(start_dim=1)
        h = torch.tanh(e @ self.W["h"].T)
        return h @ self.W["out"].T


def per_example_fitness(logits, Y):
    """Negative per-example cross-entropy. Shape (N, B) -- the Tier-0 signature."""
    N, B, V = logits.shape
    ll = -F.cross_entropy(
        logits.reshape(N * B, V), Y.repeat(N), reduction="none"
    ).reshape(N, B)
    return ll
