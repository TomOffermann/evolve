# E8 — Is Proposal B structural, or just a smaller step?

**Run:** 2026-08-13 · `code/experiments/e8_partition_control.py` · countdown post-training,
250 generations, **8 seeds** · raw output in `E8-results.txt`.

[E6](E6-results.md) found partitioned n=1 gaining +0.094 pass@16 at 3 seeds. The obvious null —
and the one that killed the last two positive results here — is that perturbing 1 of 3 parts is
just a **smaller effective step**, and E1c established that smaller steps preserve pass@k. This
run sweeps the global arm's step size across and past the partitioned arm's effective step.

## Results

| arm | pass@1 | sem | pass@16 | sem | final σ |
|---|---|---|---|---|---|
| global σ=0.0029 (norm-matched) | 0.1123 | 0.004 | 0.1567 | 0.009 | 0.0029 |
| global σ=0.005 | 0.1143 | 0.003 | 0.1401 | 0.005 | 0.0050 |
| global σ=0.015 | 0.1191 | 0.003 | 0.1562 | 0.007 | 0.0150 |
| global + resolution | 0.1172 | 0.003 | 0.1455 | 0.004 | 0.0375 |
| partitioned σ=0.005 | 0.1040 | 0.004 | 0.2017 | 0.006 | 0.0050 |
| **partitioned + resolution** | 0.0991 | 0.006 | **0.2319** | 0.012 | 0.0140 |

SFT base: pass@1 0.0547, pass@16 0.2109.

## The null is dead

**Against the best global arm across the whole sweep: pass@16 0.2319 vs 0.1567, Δ = +0.075,
+5.0 sem.**

Every global setting lands in a narrow band — 0.140 to 0.157 — regardless of step size. Sweeping
σ over a 5× range moves pass@16 by 0.017; switching to partitioned moves it by 0.075. **Step size
does not explain this**, and at matched σ=0.005 the gap is 0.2017 vs 0.1401.

This is the first result in the project to clear the 2-sem bar against a *tuned* baseline, at a
seed count that can support it.

Note also that global ES ends up **well below the SFT base** on pass@16 (0.140–0.157 vs 0.211) —
it destroys diversity to buy pass@1. Partitioned ends up **above** it (0.2319), i.e. it improved
pass@1 while leaving pass@k slightly better than where it started.

## What E6 got wrong, at 3 seeds

E6 reported "no measurable pass@1 cost". At 8 seeds there **is** one: 0.0991 vs 0.1123 for the
best global arm, ≈1.8 sem, and ≈2.8 sem against the best-pass@1 global arm (σ=0.015, 0.1191).

So this is a **trade, not a free lunch** — large gain on pass@16, real cost on pass@1. Worth
having if pass@k is what you want; not a strict improvement. E6's claim was an artefact of low
power, exactly as flagged at the time.

## The one null still standing

Partitioned has *lower* pass@1 **and** higher pass@16. A model that learns less stays closer to
its base, and the base has pass@16 0.211. "Partitioned simply learns more slowly" predicts both
observations at once.

Distinguishing them needs the whole trade-off curve, not a single point: sweep training progress
and plot pass@1 against pass@16 for both methods.

- partitioned's curve **above** global's at matched pass@1 → structural
- both on the **same** curve → partitioned is a slower knob

→ **[E9](E9-results.md)** does exactly this, using training checkpoints so one run yields a whole
curve. This is the framing arXiv:2601.20861 used for ES vs GRPO (ES traced a convex Pareto front;
GRPO sat top-right) — same question one level down.

## Reading of the σ column

The resolution rule settles at very different values per method: **0.0375 for global, 0.0140 for
partitioned.** It is not merely reproducing the fixed-σ arms, and partitioned+resolution beats
partitioned+fixed (0.2319 vs 0.2017). Whatever the rule is tracking, it tracks something
method-specific — worth a look, since it was designed only to hold the tie rate.

## Standing

Not yet promoted to a decision. E9 first. But if the frontier separates, Proposal B becomes the
first mechanism in this project to survive a properly tuned control — and it is also the one with
external support, since arXiv:2601.20861 identifies dense high-norm ES updates as the cause of
catastrophic forgetting and proposes no mitigation. See the
[evidence dossier](../literature/es-vs-rl-evidence.md).
