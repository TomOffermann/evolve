# E9 — Pareto frontier: is partitioned structural, or just slower?

**Run:** 2026-08-13 · `code/experiments/e9_pareto.py` · countdown post-training, 300 generations,
5 seeds, evaluated at 7 checkpoints · raw output in `E9-results.txt`.

[E8](E8-results.md) killed the step-size null: partitioned beats the best-tuned global arm by
+5.0 sem on pass@16. One null survived — partitioned also has *lower* pass@1, and "it simply
learns more slowly" predicts lower pass@1 and higher pass@16 **together**, since a model that
learns less stays nearer a base that has pass@16 0.211.

The test is the whole trade-off curve, not a single point. Training checkpoints give it cheaply:
one run yields an entire frontier.

## The two trajectories

| gen | global pass@1 | global pass@16 | partitioned pass@1 | partitioned pass@16 |
|---|---|---|---|---|
| 0 | 0.0750 | 0.2328 | 0.0750 | 0.2328 |
| 40 | 0.0953 | 0.2172 | 0.0828 | 0.2422 |
| 80 | 0.1078 | 0.1820 | 0.0906 | 0.2266 |
| 130 | 0.1148 | 0.1570 | 0.0898 | 0.2352 |
| 190 | 0.1164 | 0.1461 | 0.0930 | 0.2250 |
| 250 | 0.1195 | 0.1414 | 0.1039 | 0.2281 |
| 300 | 0.1195 | 0.1391 | 0.1102 | 0.2211 |

The shapes differ qualitatively, not just in speed:

- **Global's pass@16 falls monotonically** 0.233 → 0.139, losing 40% of its diversity to buy
  pass@1. This is textbook RLVR collapse.
- **Partitioned's pass@16 is essentially flat** 0.233 → 0.221, hovering at the base level
  throughout while pass@1 climbs.

## The frontier comparison

Global's pass@16 interpolated at partitioned's pass@1:

| partitioned | pass@1 | its pass@16 | global at same pass@1 | Δ |
|---|---|---|---|---|
| gen 40 | 0.0828 | 0.2422 | 0.2268 | +0.0154 |
| gen 80 | 0.0906 | 0.2266 | 0.2208 | +0.0058 |
| gen 130 | 0.0898 | 0.2352 | 0.2214 | +0.0138 |
| gen 190 | 0.0930 | 0.2250 | 0.2190 | +0.0060 |
| gen 250 | 0.1039 | 0.2281 | 0.1930 | **+0.0351** |
| gen 300 | 0.1102 | 0.2211 | 0.1737 | **+0.0474** |

**Δ is positive at every checkpoint, and it grows with training.** If partitioned were merely a
slower version of the same process, both methods would trace one curve and Δ would sit at zero
throughout. It does not.

## Verdict, with the limit stated

**The "just slower" null is rejected. The gain is structural.**

But be precise about where: **early Δ (+0.006 to +0.015) is within noise** — partitioned pass@16
sem is 0.011–0.018 — so at low pass@1 the two methods are on the same curve as far as this run can
tell. The separation appears at the pass@1 levels that actually matter: **+0.035 at pass@1 0.104
(~2 sem) and +0.047 at pass@1 0.110 (~2.6 sem)**.

The honest statement is therefore: *the frontiers coincide early and separate as training
progresses, with partitioned dominating in the range where a practitioner would stop.*

**Partitioned is genuinely slower on pass@1** — 0.1102 at generation 300 against global's 0.1195
at 250. It costs sample efficiency. What it buys is that at any given pass@1 level it retains
substantially more pass@k.

## Why this matters beyond pass@k

Global ES here reproduces exactly the failure arXiv:2601.20861 reports: it drifts from the base
and loses what the base had. Partitioned ES climbs pass@1 while staying at base-level pass@16.

That paper's diagnosis was **dense, high-norm ES updates**, with no mitigation proposed.
Partitioned perturbation is a candidate mitigation and this is the first evidence it does
something. The unresolved link is whether the mechanism is the one they name — see the caveats in
[Proposal B](../proposals/B-modular-partition-diversity.md): partitioning does **not** obviously
produce a sparse *aggregate* update, since summed over 128 members every part still receives
contributions each generation. What differs is the correlation structure of individual
perturbations, not necessarily the sparsity of ΔW.

**Measuring update norm and sparsity directly is the next thing to do**, and it is cheap.

## Limits

5 seeds, one task, 94k parameters, one partition (3 matrices). Nothing here says the effect
survives a real pretrained model, more parts, or a different task — which is exactly what the
[benchmark plan](../benchmarks/design.md) is for. The pass@1 cost is real and would matter to
anyone optimising pass@1 alone.
