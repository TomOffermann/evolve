# 0006 — Partitioned perturbation is the first mechanism to survive its controls

**Date:** 2026-08-13 · **Status:** Settled for this benchmark; open at scale.

## Decision

**Partitioned perturbation (Proposal B / P2) is adopted as the recommended sampler for
sparse-reward post-training where pass@k matters.** `PartitionedSampler(n_active=1)` with the
resolution rule.

It is the first mechanism in this project to clear a properly tuned control at a seed count that
supports it. Every previous positive result failed on exactly that.

## Evidence

**[E8](../../research/experiments/E8-results.md)** — 8 seeds, step-size null:

| arm | pass@1 | pass@16 |
|---|---|---|
| best global (σ swept 0.0029–0.015 + resolution) | 0.1123–0.1191 | 0.1401–**0.1567** |
| partitioned + resolution | 0.0991 | **0.2319** |

+0.075 pass@16, **+5.0 sem**. Sweeping σ over a 5× range moves global's pass@16 by 0.017;
partitioning moves it by 0.075.

**[E9](../../research/experiments/E9-results.md)** — 5 seeds, "just slower" null, via the whole
trade-off curve:

- Global's pass@16 falls monotonically 0.233 → 0.139 as pass@1 climbs. Textbook RLVR collapse.
- Partitioned's pass@16 stays flat at 0.233 → 0.221 while pass@1 climbs.
- At matched pass@1, Δ is positive at every checkpoint and **grows** with training: +0.035 at
  pass@1 0.104, **+0.047 at pass@1 0.110**.

A slower version of the same process would trace the same curve. It does not.

## Stated limits — this is not a free lunch

- **There is a real pass@1 cost.** 0.0991 vs 0.1123 in E8 (~1.8 sem); partitioned reaches pass@1
  0.1102 at generation 300 where global reaches 0.1195 at 250. It trades sample efficiency on
  pass@1 for retained pass@k. Anyone optimising pass@1 alone should not use it.
- **The frontiers coincide early.** Δ at low pass@1 (+0.006 to +0.015) is inside the noise. The
  domination is established in the upper pass@1 range only.
- **94k parameters, one task, one partition (3 matrices), 5–8 seeds.** Nothing here says it
  survives a real pretrained model, more parts, or another task.
- E6's "no measurable pass@1 cost" was a 3-seed artefact and is retracted.

## Why this is more than a pass@k result

arXiv:2601.20861 reports ES post-training losing ~10% held-out capability where GRPO loses none,
diagnoses **dense high-norm ES updates**, and proposes no mitigation. Global ES here reproduces
that shape — it drifts from the base and loses what the base had. Partitioned ES climbs pass@1
while holding base-level pass@16.

**The mechanism link is not established, and the obvious version of it is wrong.** Partitioning
does not produce a sparse *aggregate* update: with 128 members over 3 parts, ~43 hit each part
every generation, so ΔW is dense. Partitioning makes each *member's* perturbation sparse, not the
update.

**The working hypothesis is cross-block credit contamination.** To first order a global member's
fitness is `F(W) + σΣ_q ⟨∇_q, E_q⟩`, so the estimator for block *p* carries mean-zero but
nonzero-variance noise from every other block:

    global block-p variance      ∝ (Σ_q ‖∇_q‖²) / N
    partitioned block-p variance ∝ P·‖∇_p‖² / N

Partitioning therefore wins for block *p* exactly when `‖∇_p‖² < mean_q ‖∇_q‖²` — it should help
below-average-gradient blocks and *hurt* the dominant one. That is falsifiable, connects to
Dominant-Layer ZO, and is mildly consistent with our per-part utility trace (out 0.41, h 0.33,
emb 0.26). It explains the data; it is not verified.

**Prior art:** the operator is not novel — MeZO-BCD perturbs and updates one block per step, and
ZO-BCD (ICML 2021) predates it. What appears unexamined is its effect on *output diversity*: that
literature measures convergence, wall-clock and accuracy, and arXiv:2501.19099 proves structured
and dense perturbations have equivalent average convergence while explicitly not measuring pass@k
or which solution is reached. See [Proposal B](../../research/proposals/B-modular-partition-diversity.md).

**Next, and cheap: measure update norm and sparsity directly** for both samplers. That either
confirms the published mechanism or shows partitioning works for a different reason — both are
worth knowing, and the second would be more interesting.

## Consequences

- `PartitionedSampler` promoted in the registry from UNTESTED to recommended-with-caveats.
- The retention test (held-out capability, real pretrained base) becomes the highest-value next
  experiment — it is where this claim either generalises or dies.
- P2's other claims (per-part credit assignment, learned partitions via epistasis) remain
  untested and are now better motivated.
