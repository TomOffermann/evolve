# Proposals

Bar for entry: an intuition a reader can judge in 30 seconds, mechanics concrete enough to
implement, an honest cost, and a stated way for the idea to be **wrong**.

> **Standing decision (2026-08-13):** **[Proposal B is adopted](../../claude/decisions/0006-partitioned-perturbation-works.md)**
> for sparse-reward post-training where pass@k matters — the first mechanism here to clear a
> properly tuned control ([E8](../experiments/E8-results.md): +5.0 sem vs the best global arm
> across a 5× σ sweep; [E9](../experiments/E9-results.md): frontier dominates at matched pass@1).
> It costs ~1.8 sem of pass@1 and is measured only at 94k params on one task.
>
> **No *weighting* mechanism is established.** The Tier-0 novelty bonus beat its control by
> +2.7 sem in E4, but that control had a frozen RNG seed; against one that varies the margin is
> +1.1/+1.7 sem ([E7](../experiments/E7-results.md), decision 0005). Tier-0 as an **instrument**
> (E0) is untouched.
>
> The pattern across all of it: the mechanisms that failed all reweighted the population;
> the one that worked changed **what gets perturbed**.

## Status after [E0](../experiments/E0-results.md) (2026-08-12)

| | Proposal | Measures diversity in | Verdict | ρ vs held-out behaviour |
|---|---|---|---|---|
| **A** | [Functional Signature Diversity](A-functional-signature-diversity.md) | **function space** — what the model does | **Tier 0 validated.** Tier 1 falsified. | **0.76 – 0.87** |
| **B** | [Modular Partition Diversity](B-modular-partition-diversity.md) | **structure space** — which part was perturbed | ✅ **Validated** (E8/E9, decision 0006). Costs pass@1. | — |
| **C** | [Active-Subspace Diversity](C-active-subspace-diversity.md) | **geometry** — the dims the optimiser uses | **Falsified as a metric.** Survives as variance reduction. | 0.010 – 0.019 |

Floor (random) 0.00 · ceiling 0.76–0.87.

## Revised build order: A(Tier 0) → B → C(sampling only)

The original order was C → A → B, on the reasoning that C was cheapest and carried a guaranteed
variance-reduction payoff. E0 killed C's diversity claim outright, so:

- **A Tier 0 first.** ~20 lines, free, already validated, objective-agnostic. It becomes the
  measurement infrastructure everything else is judged against. Nothing else needs building
  before it.
- **B second**, and it is now the only substantive open bet. E0 doesn't touch it — B changes the
  *representation* of the genotype rather than the metric on it, so none of E0's negative results
  apply. It now has a validated yardstick to be judged against, which it didn't before.
- **C last, and only its sampling half** (orthogonal directions, antithetic pairs). Judge on
  progress-per-rollout, not on diversity. Keep the subspace sketch as a diagnostic only.

## Two lessons from E0 that outlive the specific proposals

**1. Project along the objective; don't sketch the outputs.** Random projection is
distance-preserving, and in output space that is a defect — it faithfully preserves the
task-irrelevant variation that dominates. Same delta, JL sketch → ρ = 0.16; read along the task
direction → ρ = 0.84.

**2. Spread is not signal.** Proposal C produced a metric with 18× the discriminative range of
naive parameter cosine and *zero* predictive power. Its histogram looked healthy. Every new
metric gets tested against a held-out behavioural target before it is allowed near selection.

## Shared machinery (build once)

- Tier-0 signature + double-centring. **First thing to land.** (Prototype:
  `code/evolve/diversity/metrics.py`.)
- `log det(K + εI)` population volume (DvD) + greedy DPP-MAP selection for `N > 10^3`.
- DEGA phase controller: exploit when fitness spread is high, diversify when it collapses.
- DiPEC subsampled-step operator (`1/λ` mask on rank-1 components).
- Thompson-sampled reward/diversity weight (from DvD) — do not hand-tune this.
- The E0 harness itself, kept as a **regression test** for any future metric.

## The question E0 did not answer — now answered

E0 established that Tier 0 *predicts behaviour*. Necessary, not sufficient. E1 asked whether
routing it into the update improves optimisation and found **no** — but on a dense-reward task
where E1e showed there was no collapse to fix, so the question was never really put.

[E4](../experiments/E4-results.md) put it properly, on sparse verifiable reward: **yes.** Tier 0
as a novelty bonus beats its matched random control by +2.7 sem on pass@16 at equal pass@1.

So Tier 0 now has **both** roles:
- **Instrument** — the collapse detector that says whether a mechanism is warranted (free, always on).
- **Mechanism** — the novelty bonus, warranted when the detector's policy-side counterpart fires.

The instrument role is what generalises; the mechanism role is conditional on the regime.
