"""The collapse detector — the one diversity component that ships.

Rationale (see claude/decisions/0002): E1-E1e found that no diversity mechanism
beats a *tuned* baseline, because on the tasks tried there is no diversity collapse
to fix. ES resamples its population from scratch every generation, so population
diversity cannot collapse the way a persistent GA's can.

That makes the Tier-0 signature validated by E0 (rho = 0.76-0.87 against held-out
behavioural decorrelation) useful in a different role than intended: not as a term
in the update, but as the **instrument that says whether a mechanism is warranted at
all**. If these numbers stay flat, adding diversity machinery is guaranteed waste.

Cost: zero extra forward passes. It reads the per-example fitness matrix the ES loop
already computes.

    det = CollapseDetector()
    ...
    f = per_example_fitness(logits, Y)     # (N, M), computed anyway
    det.observe(f, F)
    if det.fires():
        ...                                # only now is a mechanism worth trying
"""

import numpy as np
import torch


def double_centre(f):
    """Remove per-example difficulty and per-member skill. Without this you measure
    'who is good' rather than 'who is different' -- E0."""
    return f - f.mean(0, keepdim=True) - f.mean(1, keepdim=True) + f.mean()


def signature(f):
    """Tier 0: double-centred, row-normalised per-example fitness. (N, M)."""
    z = double_centre(f)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def effective_rank(S, eps=1e-12):
    """exp(entropy of the normalised singular values): how many independent
    behavioural directions the population actually spans. For N=128 members,
    E1e measured this drifting 92 -> 65 and then flattening -- i.e. no collapse."""
    s = torch.linalg.svdvals(S - S.mean(0, keepdim=True))
    p = s / s.sum().clamp_min(eps)
    return float(torch.exp(-(p * torch.log(p.clamp_min(eps))).sum()))


def log_volume(S, tau=1.0, eps=1e-6):
    """DvD population volume, log det(K + eps I), on the signature kernel."""
    d = 1.0 - S @ S.T
    K = torch.exp(-d / tau)
    K = K + eps * torch.eye(K.shape[0], device=K.device)
    return float(torch.linalg.slogdet(K)[1])


class CollapseDetector:
    """Rolling monitor. `fires()` is the gate on building any diversity mechanism.

    The trigger is a *sustained relative decline* in effective rank, compared
    against an early-training reference rather than an absolute threshold -- the
    absolute value depends on N, M and the task, so only its trajectory is portable.
    """

    def __init__(self, warmup=25, window=50, drop=0.5, subsample=256):
        self.warmup, self.window, self.drop = warmup, window, drop
        self.subsample = subsample
        self.reference = None
        self.history = []
        self.step = 0

    @torch.no_grad()
    def observe(self, f, F=None):
        """f: (N, M) per-example fitness. Returns the current reading."""
        self.step += 1
        if f.shape[0] > self.subsample:              # keep the cost O(subsample^2)
            idx = torch.randperm(f.shape[0], device=f.device)[: self.subsample]
            f = f[idx]
        S = signature(f)
        rec = {
            "step": self.step,
            "eff_rank": effective_rank(S),
            "log_volume": log_volume(S),
            "fitness_spread": float(F.std()) if F is not None else float("nan"),
        }
        self.history.append(rec)
        if self.step == self.warmup:
            self.reference = rec["eff_rank"]
        if len(self.history) > 4 * self.window:
            self.history.pop(0)
        return rec

    def fires(self):
        """True when effective rank has stayed below `drop` x its reference across a
        full window. Until this is True, a diversity mechanism is not warranted."""
        if self.reference is None or len(self.history) < self.window:
            return False
        recent = [h["eff_rank"] for h in self.history[-self.window:]]
        return max(recent) < self.drop * self.reference

    def summary(self):
        if not self.history:
            return "no observations"
        h = self.history[-1]
        ref = f"{self.reference:.1f}" if self.reference else "warming up"
        return (f"step {h['step']}  eff_rank {h['eff_rank']:.1f} (ref {ref})  "
                f"log_vol {h['log_volume']:.1f}  spread {h['fitness_spread']:.4f}  "
                f"{'COLLAPSE' if self.fires() else 'ok'}")


class PolicyDiversityDetector:
    """Watches the **policy's own output distribution**, not the ES population.

    E2 showed these are different objects and only one of them collapses. The ES
    population resamples from a fixed isotropic noise distribution every generation,
    so its diversity cannot collapse (E1e) -- `CollapseDetector` correctly reported
    'ok' on all seeds. Meanwhile the *policy* collapsed hard: pass@1 doubled while
    pass@16 fell, exactly the documented RLVR failure.

    Monitoring only the population therefore misses the failure the whole programme
    is about. Use both.

    Two readings, both cheap:
      entropy   -- mean per-token entropy of the policy's output distribution
      distinct  -- distinct sampled completions per problem, at k samples
    """

    def __init__(self, warmup=10, window=40, drop=0.6):
        self.warmup, self.window, self.drop = warmup, window, drop
        self.reference = None
        self.history = []
        self.step = 0

    @torch.no_grad()
    def observe(self, logits=None, samples=None):
        """logits: (..., V) policy output distribution over a probe set.
        samples: (k, B, L) sampled completions, for the distinct-count reading."""
        self.step += 1
        rec = {"step": self.step}
        if logits is not None:
            p = torch.softmax(logits, dim=-1)
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
            rec["entropy"] = float(ent.mean())
        if samples is not None:
            k, B, L = samples.shape
            flat = samples.permute(1, 0, 2).reshape(B, k, L)
            uniq = [len({tuple(row.tolist()) for row in flat[b]}) for b in range(B)]
            rec["distinct"] = float(np.mean(uniq)) / k
        self.history.append(rec)
        key = "entropy" if logits is not None else "distinct"
        if self.step == self.warmup:
            self.reference = rec[key]
        if len(self.history) > 4 * self.window:
            self.history.pop(0)
        return rec

    def fires(self, key="entropy"):
        if self.reference is None or len(self.history) < self.window:
            return False
        recent = [h[key] for h in self.history[-self.window:] if key in h]
        return bool(recent) and max(recent) < self.drop * self.reference

    def summary(self, key="entropy"):
        if not self.history or key not in self.history[-1]:
            return "no observations"
        h = self.history[-1]
        ref = f"{self.reference:.3f}" if self.reference else "warming up"
        return (f"step {h['step']}  {key} {h[key]:.3f} (ref {ref})  "
                f"{'COLLAPSE' if self.fires(key) else 'ok'}")
