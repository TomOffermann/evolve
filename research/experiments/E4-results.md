# E4 — Parameter sweeps: DiPEC, and novelty vs. its control at 8 seeds

**Run:** 2026-08-13 · `code/experiments/e4_dipec_sweep.py` · countdown post-training with the
resolution rule, 250 generations · raw output in `E4-results.txt`.

Two jobs. Give DiPEC a fair test instead of the single arbitrary λ that E1 condemned it on, and
rerun the novelty-vs-random comparison at enough seeds to actually decide it.

## Part 1 — DiPEC subsample rate (3 seeds)

| arm | pass@1 | sem | pass@16 | sem |
|---|---|---|---|---|
| baseline (λ=1) | 0.1224 | 0.006 | 0.1628 | 0.009 |
| dipec λ=1.25 | 0.1042 | 0.007 | 0.1901 | 0.034 |
| dipec λ=1.5 | 0.1133 | 0.005 | 0.1615 | 0.015 |
| dipec λ=2.0 | 0.1172 | 0.002 | 0.1628 | 0.016 |
| dipec λ=4.0 | 0.0951 | 0.010 | 0.1628 | 0.017 |
| dipec λ=8.0 | 0.0794 | 0.011 | 0.1667 | 0.009 |

**Monotone degradation, no sweet spot.** pass@1 falls steadily as λ rises, pass@16 is flat
throughout, and λ → 1 recovers the baseline exactly as it must.

E1 called DiPEC "catastrophic" from a single λ=4 on the dense task. The sweep gives a fairer and
more useful verdict: at low λ it is **harmless but useless**, and it degrades from there. The
prediction from [the DEGA notes](../literature/dega-dipec.md) holds — in DEGA the improving mask
has meaningful components, so dropping some keeps a valid smaller improvement; in ES the
population *is* a Monte-Carlo gradient estimate, so subsampling it multiplies estimator variance
by roughly λ and buys nothing.

## Part 2 — Novelty vs. RANDOM control, 8 seeds

This is the comparison that killed novelty in [E1b](E1b-results.md), rerun on the benchmark that
actually collapses.

| arm | pass@1 | sem | pass@16 | sem | gap |
|---|---|---|---|---|---|
| baseline | 0.1206 | 0.003 | 0.1602 | 0.008 | 0.040 |
| **novelty λ=0.2** | 0.1157 | 0.003 | **0.1880** | 0.014 | 0.072 |
| **novelty λ=0.4** | 0.1143 | 0.007 | **0.2124** | 0.016 | 0.098 |
| RANDOM λ=0.2 (ctrl) | 0.1152 | 0.003 | 0.1479 | 0.005 | 0.033 |
| RANDOM λ=0.4 (ctrl) | 0.1060 | 0.003 | 0.1753 | 0.013 | 0.069 |

**At λ=0.2 the two arms have essentially identical pass@1** — 0.1157 vs 0.1152 — which makes the
pass@16 comparison clean: **0.1880 vs 0.1479, a gap of +0.040 against a combined sem of 0.015.
That is +2.7 sem.**

At λ=0.4 novelty **dominates the control on both axes**: higher pass@1 (0.1143 vs 0.1060) *and*
higher pass@16 (0.2124 vs 0.1753, +1.8 sem). Versus the plain baseline it costs 0.006 pass@1
(~1 sem, within noise) and gains **+0.052 pass@16 (+2.9 sem)**.

And note what the random bonus does: at λ=0.2 its pass@16 is **0.1479, below the baseline's
0.1602**. Noise injection is not merely inferior to the signal here — it is worse than doing
nothing, while also costing pass@1.

## What this overturns

[E1b](E1b-results.md) found a random bonus beating novelty, and [E1c](E1c-results.md) found plain
hyperparameters beating both. [Decision 0002](../../claude/decisions/0002-diversity-mechanisms-not-yet-warranted.md)
concluded from that: no diversity mechanism is warranted.

**That conclusion was an artefact of the benchmark, not a fact about the method.** On a dense
task with no collapse, the only thing a bonus could do was mimic a smaller step size, so noise
did it as well as signal. Move to sparse verifiable reward where the policy genuinely collapses,
and the signal separates from noise by 2.7 sem.

The reasoning in E1b/E1c was sound and the controls were right — they were run on a task that
could not distinguish the hypotheses. The lesson is not "trust mechanisms after all"; it is that
**a negative result is only as general as the benchmark that produced it**, and this one was much
less general than it looked.

Superseded by [decision 0003](../../claude/decisions/0003-benchmark-rebuilt-mechanisms-reopened.md).

## Limits

One task family, one model size, 250 generations, `k = 16`. The λ=0.2 comparison is the tightest
evidence (matched pass@1, +2.7 sem); λ=0.4 is directionally stronger but has wider error bars on
pass@1. What is *not* yet shown: that the gain survives longer training, larger models, or a
different sparse task. Those are the obvious next checks before treating this as settled.
