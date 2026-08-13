"""Name -> operator, so runs can be specified by config instead of by import.

Each entry carries its measured verdict, because the whole point of E0-E4 was that
"does this operator help" has no regime-independent answer. `describe()` prints the
table; use it before reaching for something exotic.
"""

from __future__ import annotations

from .operators import sampling, sigma, weighting

WEIGHTING = {
    "rank":        (weighting.RankWeighting,   "baseline; best on dense objectives with no collapse"),
    "novelty":     (weighting.NoveltyBonus,    "NOT ESTABLISHED vs noise (E7: +1.1/+1.7 sem, under bar); helps vs no-mechanism"),
    "random":      (weighting.RandomBonus,     "CONTROL for novelty -- run in every comparison; its seed MUST vary per run"),
    "novelty_ada": (weighting.AdaptiveNovelty, "worse than fixed lam (E1); schedule ratchets up and stalls"),
    "niche":       (weighting.NicheBalanced,   "neutral (E1d: +0.004 +/- 0.006)"),
    "whitened":    (weighting.RedundancyWhitened, "worse (E1: -0.30); GLS amplifies the noisiest directions"),
    "dipec":       (weighting.DipecPartial,    "does not port to ES (E4: monotone degradation, no sweet spot)"),
}

SAMPLING = {
    "iid":         (sampling.IIDSampler,         "default; EGGROLL's own choice"),
    "antithetic":  (sampling.AntitheticSampler,  "worse than iid (E1, provisional); halves independent directions"),
    "orthogonal":  (sampling.OrthogonalSampler,  "UNTESTED; carries a lower-MSE theorem, judge on progress/rollout"),
    "partitioned": (sampling.PartitionedSampler, "UNTESTED; Proposal B / P2 -- the open research bet"),
}

SIGMA = {
    "fixed":      (sigma.FixedSigma,       "correct only if sigma has been swept"),
    "resolution": (sigma.ResolutionRule,   "RECOMMENDED default; matches a hand sweep without sweeping"),
    "onefifth":   (sigma.OneFifthRule,     "classical; INVERTS under sparse binary reward (E2)"),
    "cosine":     (sigma.CosineDecaySigma, "schedule; use as a control against adaptive rules"),
}

_TABLES = {"weighting": WEIGHTING, "sampling": SAMPLING, "sigma": SIGMA}


def build(kind: str, name: str, **kwargs):
    table = _TABLES[kind]
    if name not in table:
        raise KeyError(f"unknown {kind} '{name}'. Options: {sorted(table)}")
    return table[name][0](**kwargs)


def describe(kind: str | None = None) -> str:
    out = []
    for k, table in _TABLES.items():
        if kind and k != kind:
            continue
        out.append(f"\n{k}:")
        for name, (_, note) in table.items():
            out.append(f"  {name:12s} {note}")
    return "\n".join(out)


if __name__ == "__main__":
    print(describe())
