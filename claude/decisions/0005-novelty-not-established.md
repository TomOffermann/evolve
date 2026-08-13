# 0005 — The novelty bonus is not established; the control was broken

**Date:** 2026-08-13 · **Status:** Settled. **Supersedes
[0003](0003-benchmark-rebuilt-mechanisms-reopened.md).**

## Decision

**No diversity mechanism is recommended as established.** The default remains EGGROLL +
`ResolutionRule` for σ, with both collapse detectors logged.

The Tier-0 novelty bonus at λ=0.4 is the best measured setting for pass@k on the countdown
benchmark (pass@16 0.2007 vs 0.1479 for no mechanism) and may be used deliberately when pass@k
matters. It is **not** established as better than injecting equivalent noise.

## What changed

0003 rested on E4: novelty beating a random control by +2.7 sem. That control had a **fixed RNG
seed independent of the run seed**, so across all 8 seeds it drew the identical bonus sequence.
E4 measured one realization replicated eight times, and its reported sem of 0.005 excluded the
control's own variability.

Rerun with a control that varies ([E7](../../research/experiments/E7-results.md), 8 seeds):

| arm | pass@1 | pass@16 |
|---|---|---|
| no mechanism | 0.1172 | 0.1479 |
| novelty λ=0.2 | 0.1118 | 0.1680 |
| novelty λ=0.4 | 0.1060 | **0.2007** |
| RANDOM λ=0.2 | 0.1147 | 0.1548 |
| RANDOM λ=0.4 | 0.1182 | 0.1743 |

Margin falls from +2.7 sem to **+1.1 sem (λ=0.2) and +1.7 sem (λ=0.4)** — both under the 2-sem
bar. And at λ=0.4 novelty *trades* pass@1 for pass@16 rather than dominating, so E4's "dominates
on both axes" was the same artefact.

The control rose from 0.1479 to 0.1743 once free to vary — it had been pinned to an unluckily low
sample, and the whole margin was measured against that.

## What still stands

- **Some bonus helps pass@k.** Both arms beat no-mechanism (0.148 → 0.168/0.201 and
  0.155/0.174). Partially decoupling the update from fitness preserves pass@k.
- **Novelty is directionally ahead at both λ**, consistently, and reaches the highest pass@16
  measured. Weak positive evidence.
- **The benchmark still reproduces RLVR collapse** (pass@1 up, pass@16 down) — 0003's benchmark
  work is unaffected, only its mechanism conclusion.
- **Tier-0 as an instrument** (E0, ρ = 0.76–0.87) is a separate measurement and untouched.
- The `ResolutionRule` finding is untouched: the classical 1/5th rule inverts under sparse binary
  reward, and controlling tie rate instead matches a hand-swept σ without the sweep.

## The rule this earns

0003 gave us *a negative result is only as general as the benchmark that produced it*. This is its
twin:

> **A positive result is only as good as its control's ability to beat it.** Before believing a
> margin, check the control is free to vary on every axis the treatment varies on. A control with
> a frozen seed is not a control — it is one sample.

Three of this project's conclusions have now been overturned, and in every case the cause was a
control that could not do its job: a benchmark that could not exhibit the failure (E1), a control
that could not collapse (E3, twice), a control that could not vary (E4). The pattern is specific
enough to check for directly, and checking is cheap compared to building on it.

## Open

**n ≈ 18–20 seeds would settle it** at the observed λ=0.4 effect size — roughly 90 minutes, the
cheapest decisive experiment available. Until then the effect is neither believed nor discarded.
