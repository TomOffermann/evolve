"""Autoregressive policy with EGGROLL population rollouts.

Difference from `model.py`: in RL-style post-training the population's trajectories
**diverge** -- each member samples its own tokens, so inputs are per-member (N, B,
ctx) rather than shared (B, ctx). The low-rank factorisation still applies at every
layer; what is lost is sharing the base matmul across the population, which is
inherent to ES-for-RL and not something EGGROLL claims to avoid.

Reward is sparse and binary, so per-example fitness is a **bit vector** -- exactly
the regime E0 flagged as the highest-value follow-up for the Tier-0 signature.
"""

import numpy as np
import torch
import torch.nn.functional as F


class TinyPolicy:
    def __init__(self, vocab, ctx, emb=32, hidden=192, device="cpu", seed=0):
        g = torch.Generator().manual_seed(seed)
        self.vocab, self.ctx, self.emb, self.hidden = vocab, ctx, emb, hidden
        self.device = device
        self.W = {
            "emb": torch.randn(vocab, emb, generator=g) * 0.3,
            "h": torch.randn(hidden, ctx * emb, generator=g) / (ctx * emb) ** 0.5,
            "out": torch.randn(vocab, hidden, generator=g) / hidden**0.5,
        }
        self.W = {k: v.to(device) for k, v in self.W.items()}

    @property
    def shapes(self):
        return {k: tuple(v.shape) for k, v in self.W.items()}

    def n_params(self):
        return sum(v.numel() for v in self.W.values())

    # ---------------------------------------------------------------- base model

    def logits_base(self, X):
        """X: (B, ctx) -> (B, vocab)."""
        e = self.W["emb"][X].flatten(start_dim=1)
        return torch.tanh(e @ self.W["h"].T) @ self.W["out"].T

    # ------------------------------------------------------------ population pass

    def logits_pop(self, X, fac, sigma, rank):
        """X: (N, B, ctx) int64 -> (N, B, vocab). Perturbation E = (1/sqrt(r)) A B^T."""
        N, B, C = X.shape
        s = sigma / rank**0.5
        A, Bf = fac["emb"]                                   # (N,vocab,r), (N,emb,r)
        e0 = self.W["emb"][X]                                # (N, B, ctx, emb)
        # Per-member row lookup into each member's own perturbation factor.
        idx = X.reshape(N, B * C, 1).expand(N, B * C, A.shape[-1])
        a_rows = torch.gather(A, 1, idx).reshape(N, B, C, A.shape[-1])
        e = e0 + s * torch.einsum("nbcr,ndr->nbcd", a_rows, Bf)
        h_in = e.flatten(start_dim=2)                        # (N, B, ctx*emb)

        A, Bf = fac["h"]
        h = torch.tanh(h_in @ self.W["h"].T
                       + s * torch.einsum("nbi,nir,nhr->nbh", h_in, Bf, A))

        A, Bf = fac["out"]
        return h @ self.W["out"].T + s * torch.einsum("nbi,nir,nvr->nbv", h, Bf, A)

    # ------------------------------------------------------------------- rollouts

    @torch.no_grad()
    def rollout(self, ctx_tokens, out_len, fac=None, sigma=0.0, rank=1,
                temperature=1.0, n_pop=1, generator=None, common_randoms=True):
        """ctx_tokens: (B, 4) problem context. Returns emitted tokens (N, B, out_len).

        Sampling is stochastic -- that is the point. Post-training has to move the
        *distribution*, and pass@k only means anything if the policy samples.

        `common_randoms` shares the sampling randomness across the whole population
        (inverse-CDF sampling from one shared uniform draw) instead of letting each
        member roll its own dice. This matters enormously here. Member fitness is a
        mean over B Bernoulli rollouts at p ~ 0.07, so independent sampling gives it
        a standard deviation of ~0.026 -- far larger than the fitness differences a
        small weight perturbation actually produces. Without common random numbers
        the ES gradient is estimated almost entirely from sampling noise, and
        training goes backwards. With them, member differences reflect weight
        differences, which is what we are trying to measure.

        Set False for *evaluation* (pass@k needs genuinely independent samples)."""
        B = ctx_tokens.shape[0]
        N = n_pop
        seq = torch.zeros(N, B, self.ctx, dtype=torch.long, device=self.device)
        seq[:, :, : ctx_tokens.shape[1]] = ctx_tokens.unsqueeze(0)
        start = ctx_tokens.shape[1]
        out = torch.zeros(N, B, out_len, dtype=torch.long, device=self.device)

        for t in range(out_len):
            lg = (self.logits_pop(seq, fac, sigma, rank) if fac is not None
                  else self.logits_base(seq[0]).unsqueeze(0).expand(N, -1, -1))
            p = F.softmax(lg / temperature, dim=-1)
            if common_randoms:
                u = torch.rand(1, B, 1, generator=generator, device=self.device)
                cdf = p.cumsum(-1)
                cdf[..., -1] = 1.0
                tok = torch.searchsorted(cdf.contiguous(), u.expand(N, B, 1))
                tok = tok.squeeze(-1).clamp_(max=self.vocab - 1)
            else:
                tok = torch.multinomial(p.reshape(-1, self.vocab), 1,
                                        generator=generator).reshape(N, B)
            out[:, :, t] = tok
            seq[:, :, start + t] = tok
        return out


def rewards_for(task, problems, out):
    """out: (N, B, 5) tokens -> (N, B) float32 rewards. This is the Tier-0 signature
    in its binary form."""
    N, B, _ = out.shape
    o = out.cpu().numpy()
    R = np.stack([task.reward(problems, o[i]) for i in range(N)])
    return torch.from_numpy(R)


def pass_at_k(task, problems, policy, k, ctx_tokens, temperature=1.0, generator=None):
    """Fraction of problems solved at least once in k samples. The diversity metric:
    a policy that collapses onto one answer has pass@k ~ pass@1."""
    out = policy.rollout(ctx_tokens, task.out_len, n_pop=k, temperature=temperature,
                         generator=generator, common_randoms=False)
    R = rewards_for(task, problems, out)          # (k, B)
    return float((R.max(dim=0).values > 0).float().mean()), float(R.mean())
