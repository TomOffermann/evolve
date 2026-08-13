# E7 — Novelty vs a properly randomised control, 8 seeds

**Run:** 2026-08-13 · `code/experiments/e7_novelty_power.py` · framework, countdown
post-training, 250 generations, 8 seeds · raw output in `E7-results.txt`.

## Why this run exists

[E5](E5-results.md)'s 3-seed regression came out the other way from
[E4](E4-results.md) on the novelty-vs-control comparison. Chasing *why* found a real
methodological bug rather than noise:

> **The random control used a fixed RNG seed (1234), independent of the run seed.** Across E4's
> 8 seeds it drew the *identical bonus sequence* every time. E4 measured **one realization of the
> random bonus, replicated eight times**, and reported a between-seed sem (0.005) that excluded
> the control's own variability entirely.

Fixed — `RandomBonus` now derives its RNG from `state["run_seed"]`, which the trainer seeds. This
run is the comparison redone against a control that actually varies.

## Results

| arm | pass@1 | sem | pass@16 | sem |
|---|---|---|---|---|
| no mechanism | 0.1172 | 0.003 | 0.1479 | 0.005 |
| novelty λ=0.2 | 0.1118 | 0.005 | 0.1680 | 0.009 |
| **novelty λ=0.4** | 0.1060 | 0.005 | **0.2007** | 0.013 |
| RANDOM λ=0.2 (ctrl) | 0.1147 | 0.004 | 0.1548 | 0.007 |
| RANDOM λ=0.4 (ctrl) | 0.1182 | 0.004 | 0.1743 | 0.008 |

Novelty vs its matched control:

| λ | pass@16 | Δ | | pass@1 |
|---|---|---|---|---|
| 0.2 | 0.1680 vs 0.1548 | +0.0132 | **+1.1 sem** | 0.1118 vs 0.1147 |
| 0.4 | 0.2007 vs 0.1743 | +0.0264 | **+1.7 sem** | 0.1060 vs 0.1182 |

## Verdict: decision 0003 does not survive

**E4's +2.7 sem falls to +1.1 / +1.7 sem** once the control is properly randomised. Both are
under the 2-sem bar. And at λ=0.4 novelty no longer *dominates* the control — it **trades**:
+0.026 pass@16 for −0.012 pass@1 (~2 sem). E4's claim that it "dominates on both axes" was an
artefact of the same broken control.

The control moved exactly as predicted: RANDOM λ=0.4 rose from E4's 0.1479 to 0.1743 once it was
allowed to vary. E4's control had been sitting on an unluckily low single realization, and
novelty's advantage was measured against that.

## What survives

- **Some bonus helps.** Both arms beat no-mechanism on pass@16 (0.148 → 0.168/0.201 for novelty,
  → 0.155/0.174 for random). Partially decoupling the update from fitness does preserve pass@k.
- **Novelty is directionally ahead at both λ tested**, consistently, and it reaches the highest
  pass@16 of any arm (0.2007). Two independent λ values pointing the same way is weak positive
  evidence — not a result, but not nothing either.
- **The Tier-0 signature as an instrument is untouched.** E0's ρ = 0.76–0.87 was a different
  measurement with a different control and this does not bear on it.

## What this costs the programme

This is the **third overturned conclusion**, and the second one where the cause was a control that
could not do its job. Running total:

| # | claim | why it fell |
|---|---|---|
| 1 | E1: "no mechanism helps" | benchmark could not exhibit the failure |
| 2 | E3: "ES avoids collapse" (twice) | control could not collapse — two different accidental fixes |
| 3 | E4/0003: "novelty beats noise, +2.7 sem" | control could not vary |

The rule earned in 0003 — *a negative result is only as general as the benchmark that produced
it* — now has a positive-result twin:

> **A positive result is only as good as its control's ability to beat it.** Before believing a
> margin, check that the control is free to vary on every axis the treatment varies on. A control
> with a frozen seed is not a control; it is one sample.

## Consequences

- Decision 0003 → superseded by
  [0005](../../claude/decisions/0005-novelty-not-established.md).
- `NoveltyBonus` docstring and the registry entry corrected — they claimed +2.7 sem and now say
  NOT ESTABLISHED. Code that asserts a wrong number is worse than code that asserts nothing.
- λ=0.4 pass@16 0.2007 remains the best measured setting on this benchmark; use it if you want
  pass@k, but know that a random bonus gets most of the way there.

## What would settle it

The margin is +1.7 sem at λ=0.4 with n=8. Reaching 2.5 sem at that effect size needs roughly
**n ≈ 18–20 seeds** — about 90 minutes of compute, and the cheapest decisive thing available.
Worth doing before either believing or discarding the effect.
