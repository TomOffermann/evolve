# E5 — Framework regression against E2/E4

**Run:** 2026-08-13 · `code/experiments/e5_framework_regression.py` · 3 seeds · raw output in
`E5-results.txt`.

A rewrite that changes the numbers is not a refactor but a new experiment with unknown
provenance. Before the framework was used for anything it had to land on the E2/E4 reference
points.

## Results

| arm | pass@1 | sem | pass@16 | sem |
|---|---|---|---|---|
| resolution (no mechanism) | 0.1237 | 0.006 | 0.1406 | 0.006 |
| + novelty λ=0.2 | 0.1120 | 0.011 | 0.1810 | 0.022 |
| + RANDOM λ=0.2 (ctrl) | 0.1198 | 0.009 | 0.2240 | 0.039 |

SFT base: pass@1 0.0547, pass@16 0.2109. E2 reference: resolution 0.1237 / 0.1641.

## Verdict

**The framework reproduces.** `resolution` lands at pass@1 **0.1237** against E2's **0.1237**,
collapse is reproduced (pass@16 0.141 vs base 0.211), and novelty holds pass@16 above
no-mechanism. Exact equality was never expected — the framework keys noise on global member index
where the old code keyed it per generation, so populations differ member for member.

**Chunking is exactly invariant end-to-end**: unchunked and `chunk_size=32` give identical pass@1
and pass@16. That property is what makes large sharded runs trustworthy.

## The check that failed, and what it was worth

`novelty beats its RANDOM control` came back **False** (0.1810 vs 0.2240) — opposite to E4.

Taken alone this was uninformative: the difference is 0.043 against a combined sem of 0.045, and
3 seeds is fine for "does collapse reproduce" but hopeless for re-deciding E4. **But the pattern
was diagnostic.** Novelty replicated almost exactly across two independent implementations
(0.1880 → 0.1810, sd 0.040 → 0.038); only the control moved, and its spread differed by a factor
of five between runs.

Chasing that discrepancy found the real bug: **the random control used a fixed RNG seed**, so
every run drew the identical bonus sequence. That invalidated E4's headline and led to
[E7](E7-results.md) and [decision 0005](../../claude/decisions/0005-novelty-not-established.md).

Worth recording as method: an underpowered replication is not useless if you read *which* arm
moved rather than only whether the sign flipped.

## Bugs this run's design caught

Two, both silent, both now regression-tested (`code/tests/test_framework.py`):

1. **Noise keyed on chunk id**, so `chunk_size` silently changed which perturbation each member
   drew — a sharded run could not be validated against a local one, and no fitness curve would
   show it. Fixed by keying on global member index.
2. **The countdown objective drew per-generation problems from a stateful RNG.** Under the thread
   executor two chunks could both miss the cache, both advance it, and be scored on *different
   problem sets* — so the ES update was computed across populations never evaluated on the same
   task. Cost: pass@1 0.117 → 0.020. Fixed by deriving problems from the generation number.

The second was invisible to a stateless test objective, so the suite now carries a deliberately
stateful one.
