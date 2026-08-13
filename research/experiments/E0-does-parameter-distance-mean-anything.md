# E0 — Does parameter-space distance mean anything at all?

**Status:** ✅ **Run 2026-08-12 → [results](E0-results.md).** Code: `code/experiments/e0_metric_sanity.py`.

Outcome in one line: H1 and H2 confirmed (concentration matched theory to three significant
figures); H3 confirmed for Tier 0 only, which falsified parts of Proposals A and C and produced
one finding that was in neither — *project along the objective, don't sketch the outputs*.

The design below is as originally written, with one correction marked inline that mattered a
great deal.

## Why

[P1](../problems/P1-continuous-diversity.md) claims naive parameter distance is dead by
concentration of measure. That claim is doing a *lot* of load-bearing work in this repo — all
three proposals are built on it. It is also trivially testable. Test it before building on it.

## Hypotheses

- **H1 (concentration).** Pairwise cosine similarity between independently sampled EGGROLL
  perturbations is `≈ N(0, 1/d_eff)`. The histogram is a spike; the distance matrix carries no
  usable ranking. *Predicted: confirmed, boringly.*
- **H2 (the one that matters).** Parameter-space distance does **not** predict behavioural
  decorrelation. Members that are "far apart" in weight space are no more likely to succeed and
  fail on different examples than members that are "close".
- **H3 (the alternative).** Functional signatures ([Proposal A](../proposals/A-functional-signature-diversity.md)
  Tier 0/1) and active-subspace projections ([Proposal C](../proposals/C-active-subspace-diversity.md))
  *do* predict behavioural decorrelation.

## Setup

Baseline EGGROLL, small model, any task with per-example scoring (a small LM on a character task
is fine — it must be runnable on the MacBook, see [P3](../problems/P3-objectives-and-hardware.md)).
`N ≈ 256` so that `O(N²)` is still affordable and we can compute everything exactly.

Per generation, log for every member `i`:

1. `F_i` — scalar fitness
2. `f_i ∈ R^M` — **per-example** fitness (this is Proposal A Tier 0, and it's free)
3. `cos(E_i, E_j)` — full pairwise matrix, computed from `A_i, B_i` without densifying
4. `φ_i` — delta-activation signature (Proposal A Tier 1)
5. `c_i = U_tᵀ E_i` — active-subspace coordinates (Proposal C), plus the singular spectrum of `U_t`
6. per-layer energy profile `(‖E_i^(l)‖)_l` (idea #5 — cheap, might surprise us)

> **Correction made while running (2026-08-12).** As written above, this design is circular:
> the target is per-example fitness and so is candidate metric (2). Fixed by splitting the data —
> metrics see **probe batch P**, the target is measured on **held-out batch Q**. That is also the
> stronger question: does a signature computed on one batch predict behavioural disagreement on
> data it has never seen? Second addition: a **RANDOM floor** and a **CEILING** (the same
> behavioural statistic computed on P). Without those, a ρ of 0.35 is uninterpretable — and in
> the event the ceiling turned out to be 0.87, which changed how every other row read.

## The measurement

The operational definition of a good diversity score from P1: **distant members should have
decorrelated behaviour.** So for each candidate metric `D`, compute the rank correlation between

```
D(i, j)        and        1 − corr(f_i, f_j)
```

over all pairs. A metric that works has a clearly positive correlation. A dead metric gives zero.

Also report, for each metric: the histogram of pairwise values (is it a spike?), and the effective
rank of the population's embedding matrix.

## Decision rule

| Outcome | What we do |
|---|---|
| H1 + H2 confirmed, H3 confirmed | Proceed as planned, C → A → B. |
| H1 confirmed but H3 *also* fails | The problem is harder than assumed — behaviour on the probe batch may be too coarse. Escalate to Proposal A Tier 2 and revisit before building anything. |
| H1 **rejected** (cosine carries signal) | Rewrite P1. Cheap classical niching may be back on the table, which would be excellent news and a much shorter road. |
| Per-layer energy profile (6) predicts well | Strong early evidence for [P2](../problems/P2-structured-perturbation.md)/Proposal B — structure matters more than direction. Consider reordering the build. |

## Note

Every quantity here is instrumentation on an *unmodified* EGGROLL run. No new algorithm is
required to run E0. That is deliberate: it is the cheapest possible way to find out whether the
premise of this whole research direction holds.
