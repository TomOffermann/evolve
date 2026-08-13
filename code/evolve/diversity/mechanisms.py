"""Ways to inject diversity into the EGGROLL update.

[P1](research/problems/P1-continuous-diversity.md) names three injection points:
*sampling*, *fitness shaping*, and *selection/weighting*. This module implements one
or more mechanisms at each, so they can be benchmarked head to head (E1).

All of them consume the Tier-0 signature validated by E0 -- the double-centred
per-example fitness matrix `f` (N, M), which is computed anyway and costs nothing.
Every mechanism returns the per-member weight vector `w` used in

    dW = (alpha / (N sigma)) * sum_i w_i A_i B_i^T

so they are drop-in replacements for `rank_normalise(F)`.
"""

import torch

from .metrics import double_centre


def _signature(f):
    """Tier 0: double-centred per-example fitness, row-normalised. (N, M)."""
    z = double_centre(f)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _rank(x):
    """Centred rank transform, ties averaged. See es/eggroll.rank_normalise --
    breaking ties by argsort order injects noise under sparse binary reward."""
    from ..es.eggroll import rank_normalise
    return rank_normalise(x)


# --------------------------------------------------------------------- baseline

def baseline(F, f, **kw):
    """Vanilla EGGROLL: centred rank transform of fitness. No diversity."""
    return _rank(F)


# ------------------------------------------------------- fitness shaping (NS-ES)

def novelty_bonus(F, f, lam=0.3, k=10, **kw):
    """NS-ES / NSR-ES: add a k-NN novelty bonus in signature space.

    novelty_i = mean distance to the k nearest other members. Members doing
    something unusual get their weight boosted."""
    S = _signature(f)
    d = 1.0 - S @ S.T
    d.fill_diagonal_(float("inf"))
    nov = d.topk(k, largest=False).values.mean(dim=1)
    return (1 - lam) * _rank(F) + lam * _rank(nov)


def random_bonus(F, f, lam=0.3, state=None, **kw):
    """CONTROL for `novelty_bonus`: identical mixing weight, but the bonus is pure
    noise instead of a novelty signal.

    Without this arm, "novelty helps" is unfalsifiable -- any mechanism that
    partially decouples the update from fitness injects exploration, and that alone
    can rescue a rare mode. This isolates whether the *signal* matters."""
    g = state.setdefault("rng", torch.Generator().manual_seed(1234))
    r = torch.rand(F.shape, generator=g, device=F.device)
    return (1 - lam) * _rank(F) + lam * _rank(r)


def novelty_adaptive(F, f, k=10, state=None, **kw):
    """NSRA-ES: same, but raise lambda while stagnating and lower it on progress."""
    lam = state.get("lam", 0.2)
    best, prev = F.max().item(), state.get("best", -1e9)
    if best > prev + 1e-4:
        lam, state["best"] = max(0.0, lam - 0.05), best
    else:
        lam = min(0.8, lam + 0.05)
    state["lam"] = lam
    return novelty_bonus(F, f, lam=lam, k=k)


# -------------------------------------------------------- selection / weighting

def dpp_marginal(F, f, lam=0.3, **kw):
    """DvD-flavoured: score each member by its marginal contribution to the
    population's volume, log det K - log det K_{-i}. That marginal gain is the
    diagonal of the inverse kernel (up to a constant), so it is one solve."""
    S = _signature(f)
    K = S @ S.T + 1e-3 * torch.eye(S.shape[0], device=S.device)
    marginal = -torch.log(torch.diagonal(torch.linalg.inv(K)).clamp_min(1e-12))
    return (1 - lam) * _rank(F) + lam * _rank(-marginal)


def redundancy_whitened(F, f, ridge=0.1, **kw):
    """Treat the ES update as an average over *correlated* observations.

    Members with correlated behavioural signatures carry correlated fitness noise,
    and plain ES double-counts them. Whitening the weights by the signature
    correlation matrix is generalised least squares:  w <- (K + ridge I)^-1 w.

    Redundant members get their vote shrunk; behaviourally unusual members get
    theirs amplified. This is diversity as *variance reduction* rather than as
    exploration -- one vote per behavioural direction, not per member."""
    S = _signature(f)
    N = S.shape[0]
    K = S @ S.T + ridge * torch.eye(N, device=S.device)
    w = torch.linalg.solve(K, _rank(F))
    return w / w.norm().clamp_min(1e-12) * N**0.5 * 0.29  # match baseline scale


# ------------------------------------------------------------ DEGA transplants

def dega_phases(F, f, k=10, quantile=0.4, state=None, **kw):
    """DEGA's phase structure: exploit when the population disagrees on fitness,
    diversify when fitness has flattened out (the continuous analogue of a tie)."""
    spread = F.std().item()
    hist = state.setdefault("spread", [])
    hist.append(spread)
    if len(hist) > 50:
        hist.pop(0)
    thresh = sorted(hist)[int(quantile * (len(hist) - 1))]
    if spread <= thresh and len(hist) > 5:
        state["diversity_steps"] = state.get("diversity_steps", 0) + 1
        return novelty_bonus(F, f, lam=0.8, k=k)
    return _rank(F)


def dipec_partial(F, f, lam=4, state=None, **kw):
    """DiPEC's subsampled improving step: keep a random 1/lam fraction of the
    contributing members, rescaled to preserve step size. Buys fitness in small
    targeted increments instead of collapsing onto the whole improving direction."""
    w = _rank(F)
    g = state.setdefault("gen", torch.Generator().manual_seed(0))
    keep = (torch.rand(w.shape, generator=g, device=w.device) < 1.0 / lam).to(w.dtype)
    return w * keep * lam


def niche_balanced(F, f, n_niches=8, hard=False, **kw):
    """One vote per behavioural NICHE, instead of one vote per member.

    This is the injection point E1 never tested. A fitness bonus (novelty, DvD) is
    the weakest of the three in P1 -- it biases the gradient and needs a schedule,
    and the schedule is where those methods died. This is *selection* instead.

    The actual failure on a skewed task is not that rare-mode members are missing;
    it is that they are outvoted. If 3% of the data is mode D, then members who are
    relatively good at D are a small minority of the population and their signal is
    swamped in the mean. Balancing the vote across behavioural clusters fixes
    exactly that, and nothing else.

    k-means on the Tier-0 signature, then normalise weights within each cluster so
    every cluster contributes equal total magnitude to the update."""
    S = _signature(f)
    N = S.shape[0]
    k = min(n_niches, N)

    # k-means++-ish init, then a few Lloyd steps on the unit sphere (cosine).
    idx = [int(torch.argmax(S.norm(dim=1)))]
    for _ in range(k - 1):
        d = 1.0 - (S @ S[idx].T).max(dim=1).values
        idx.append(int(torch.argmax(d)))
    C = S[idx].clone()
    for _ in range(5):
        assign = (S @ C.T).argmax(dim=1)
        for c in range(k):
            m = assign == c
            if m.any():
                C[c] = S[m].mean(0)
        C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)

    w = _rank(F)
    if hard:
        # Every niche contributes equal total magnitude. Maximally aggressive: it
        # also destroys the *between*-niche fitness ranking, so a niche of uniformly
        # bad members votes as loudly as a niche of good ones. Measured: much worse.
        out = torch.zeros_like(w)
        for c in range(k):
            m = assign == c
            if not m.any():
                continue
            wc = w[m] - w[m].mean()
            n = wc.norm()
            if n > 1e-9:
                out[m] = wc / n
    else:
        # Inverse-propensity: keep the global ranking, but scale each member's vote
        # by 1/sqrt(niche size). Members in a rare behavioural cluster are exactly
        # the ones being outvoted, and this compensates in proportion without
        # throwing the fitness signal away.
        counts = torch.bincount(assign, minlength=k).clamp_min(1).to(w.dtype)
        out = w / counts[assign].sqrt()

    return out / out.norm().clamp_min(1e-12) * (N**0.5 * 0.29)  # match baseline scale


MECHANISMS = {
    "baseline":       baseline,
    "antithetic":     baseline,   # differs in *sampling*, not weighting -- see e1
    "novelty":        novelty_bonus,
    "novelty_adapt":  novelty_adaptive,
    "dpp_marginal":   dpp_marginal,
    "whitened":       redundancy_whitened,
    "dega_phases":    dega_phases,
    "dipec":          dipec_partial,
}
