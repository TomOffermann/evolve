# 0001 — EGGROLL as substrate, DEGA as template

**Date:** 2026-08-12 · **Status:** Settled

## Decision

The evolutionary machinery in Evolve is built on **EGGROLL** (low-rank ES, arXiv:2511.16652) as
the optimizer substrate, and takes **DEGA/DiPEC** (arXiv:2507.01524) as the conceptual template
for the diversity mechanism.

## Why EGGROLL

Every previous attempt at ES on real models died on GPU throughput — naive batched ES runs at
about 1% of pure inference throughput because each member needs its own full-rank matmul.
EGGROLL's rank-`r` perturbation trick recovers 91% of inference throughput while still producing
full-rank aggregate updates. That is the difference between "ES is a nice idea" and "we can
actually run this."

Three secondary properties matter as much as the speed:

- **Inference-only.** No backward pass, no activation storage, no optimizer state. This is what
  makes weak/old hardware viable rather than a compromise.
- **Members are seeds.** Workers exchange `(seed, fitness)` — a few bytes. A heterogeneous pile of
  old GPUs on a mediocre interconnect is nearly the best case for ES and nearly the worst case for
  data-parallel SGD.
- **Objective-agnostic.** Needs a scalar per member. Opens up non-differentiable objectives,
  which is where the interesting research is.

## Why DEGA

It solves the exact tension we'll hit: exploiting good solutions destroys the diversity that made
them findable. Its answer — alternate phases, and take *subsampled* improving steps rather than
whole ones — is mechanism design we can port. It also has a real theorem behind it
(`O(n^{5/3} log^{2/3} n)` on LeadingOnes, breaking the `Θ(n²)` barrier), which is rare in this
corner of the field.

## What we are explicitly *not* taking from DEGA

Its diversity metric. Hamming distance on bitstrings does not survive the move to Gaussian
perturbations in 10^9 dimensions — see [P1](../../research/problems/P1-continuous-diversity.md).
Finding the replacement is the central open problem, not an implementation detail.

We are also not taking its guarantees. `(2+1)` on LeadingOnes tells us nothing about `N = 10^5`
on a transformer. We borrow the mechanism, not the theorem, and should say so in writing.

## Consequences

- Any diversity mechanism must be computable from low-rank factors or rollout by-products (C1),
  and must avoid `O(N²)` on the hot path (C2).
- Reuse `eggroll-es` as a reference implementation; own the perturbation/aggregation layer.
- Research notes live in top-level `research/`, not `claude/research/` — one location, not two.
  (`claude/rules.md` updated accordingly.)
