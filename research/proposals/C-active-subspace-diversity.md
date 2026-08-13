# Proposal C — Active-Subspace Diversity (ASD)

**Status:** **Falsified as a diversity metric** by [E0](../experiments/E0-results.md)
(ρ = 0.010–0.019 at two scales, against a ceiling of 0.76–0.87).
**Still open as a variance-reduction method** — that claim was never tested and stands.
**Injection points:** sampling (the surviving part), ~~fitness shaping~~.

> **E0 verdict (2026-08-12).** The premise below is wrong, and it is worth being precise about
> which part. C requires that a perturbation's *behavioural* effect be dominated by its
> active-subspace component. It is not: the `E_i` are isotropic, so their projections onto any
> fixed `k`-dim basis are just random `k`-dim vectors. Measured ρ ≈ 0.015 at both 8k and 81k
> parameters — not a scale artifact, not a warm-up artifact.
>
> **The instructive part is *how* it fails.** C's spread is 0.20 — eighteen times wider than raw
> parameter cosine's 0.011. It discriminates beautifully. It discriminates on nothing.
>
> > **Spread is not signal.** A metric that varies a lot and means nothing is more dangerous
> > than one that doesn't vary, because it looks like it is working. If we had shipped C
> > without E0's held-out target, its healthy-looking histogram would have convinced us.
>
> **What survives:** the orthogonal-sampling MSE result (§"The bonus", below) is a theorem
> about estimator variance, not about diversity, and E0 did not test it. That is now the only
> reason to build C, and it is still a good one — see the revised recommendation at the end.

## The intuitive idea

Parameter distance is meaningless because you're measuring in 10^9 dimensions, where everything
is orthogonal to everything. But **the optimiser never visits 10^9 dimensions.** Over any recent
window it has moved inside a subspace of maybe 16–64 dimensions — the span of its own recent
updates.

So project every perturbation onto *that* subspace, and distance becomes meaningful again.
Two members are "the same" if they push along the same directions the optimiser has been using;
a member is "novel" to the extent it has energy **orthogonal** to everything the optimiser has
done lately.

You get two numbers per member instead of one:
- **how much it repeats** what we're already doing (component inside the active subspace), and
- **how much it proposes something genuinely new** (orthogonal residual).

That is exploitation and exploration, measured, per member, from linear algebra we already have.

**One-line judgement:** don't measure diversity in the space of all possible weights — measure it
in the space the optimiser is actually moving through. Everything else is noise by construction.

## Mechanics

### 1. Maintain the active subspace

Keep a rank-`k` orthonormal sketch `U_t ∈ R^{d×k}`, `k ≈ 16–64`, of recent updates
`ΔW_{t−1}, …, ΔW_{t−H}`. Use frequent-directions or incremental SVD with a forgetting factor.

Cheap because each `ΔW` is *already* a sum of rank-1 terms — you never form a dense `d×d` anything.
In practice keep `U` per weight matrix, not globally.

### 2. Project each member (without materialising `E_i`)

```
c_i = U_tᵀ E_i = (1/√r) (U_tᵀ A_i) B_iᵀ        ∈ R^{k×n}, then flatten/sketch to R^k
```

`U_tᵀ A_i` is a `k×r` product — trivial. Satisfies C1: no dense delta is ever formed.

### 3. Decompose into exploitation and novelty

```
exploit_i = ‖c_i‖                        # energy inside the active subspace
novel_i   = ‖E_i‖_F² − ‖c_i‖²            # orthogonal residual
D_i       = spread of normalised c_i  +  β · novel_i
```

Population score: `log det(Gram(ĉ_1..ĉ_N) + εI)` — the DvD volume again, but on `k`-dim
coordinates that cost nothing to compute.

### 4. The bandit (this is ASEBO)

Trade sampling **inside** the active subspace (fast progress along known-good directions) against
sampling **orthogonal** to it (escape, exploration). ASEBO (Choromanski et al., NeurIPS 2019)
already does exactly this with contextual bandits and compressed sensing, and reports better
sample efficiency than SOTA blackbox optimisers. We're adding the diversity read-out and the
DEGA phase structure on top.

## The bonus that makes this worth building first

Within the active subspace, **orthogonalising the sampled directions strictly reduces the MSE of
the gradient estimator** compared to iid sampling. That's Choromanski's structured-evolution
result — a theorem, not a heuristic, and EGGROLL currently samples iid.

So Proposal C is the one case on this list where the diversity mechanism **pays for itself even
when exploration doesn't matter.** Worst case, it's a variance-reduction upgrade to the baseline
optimiser. That's a good worst case.

While you're in there: EGGROLL also doesn't use **antithetic (mirrored) pairs** `±E_i`, the
classic ES control variate. Free, five lines, test it in the same session.

## The DEGA transplant

| DEGA | ASD |
|---|---|
| Exploitation phase | Sample inside the active subspace. Fast progress. |
| Diversity phase | Sample orthogonal to it, orthogonalised among themselves. Maximum coverage of unexplored geometry. |
| Max-Hamming tie-break | Max `log det` of the `c_i` Gram among tied members. |
| DiPEC `1/λ` subsampling | Keep a random `1/λ` subset of the rank-1 components of the accepted update. |

## Why it should work

- **Cheapest of the three.** Pure linear algebra on factors we already have. No probe batch, no
  extra forward passes, no task-specific descriptor.
- **Fully objective-agnostic.** RL, LM, JEPA — identical code. Matters for
  [P3](../problems/P3-objectives-and-hardware.md).
- Solid prior art: ASEBO, Self-Guided ES (IJCAI 2020), structured/orthogonal MC.
- Gives a genuinely useful diagnostic even if the diversity mechanism is a wash: the effective
  dimensionality of `U_t` over training tells you how many directions the optimiser is actually
  using at any moment. Nobody plots this. It would be interesting.

## How it could fail

- **It's still a parameter-space proxy.** Two members orthogonal in `U_t` can be functionally
  identical (permutation and scale symmetries survive the projection). This is the honest
  weakness, and the reason [Proposal A](A-functional-signature-diversity.md) is the *primary*
  metric and C is the cheap one. Mitigation: validate C's ranking against A's on the same
  population — if they disagree, trust A.
- **Sketch drift.** `U_t` lags the optimiser; needs a forgetting factor, which is a hyperparameter,
  which is a place to lose a week.
- **The subspace may be uninteresting early.** At initialisation `ΔW` is close to noise and `U_t`
  spans nothing meaningful. Expect a warm-up period; don't judge the method during it.
- **`k` too small** → everything projects to the same place and we've reinvented the concentration
  problem in miniature. Track the singular value spectrum of the sketch and set `k` from it.

## First experiment — **done, and it went badly**

[E0 results](../experiments/E0-results.md), Finding 3. The diversity claim is dead.

## Revised recommendation

Build only the sampling half, and judge it on **wall-clock progress per rollout**, not on any
diversity criterion:

- (a) orthogonal vs iid sampling of `A_i` — does Choromanski's MSE result hold in our setting?
- (b) antithetic pairs `±E_i` on/off — EGGROLL doesn't use them; five lines.

Both are variance-reduction questions with no diversity machinery involved, both are cheap, and
both are worth knowing regardless. Keep the subspace sketch as a **diagnostic** — the effective
dimensionality of `U_t` over training is still an interesting thing nobody plots — but do not
route selection or fitness through it.
