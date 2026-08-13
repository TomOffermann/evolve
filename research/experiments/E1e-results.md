# E1e — Was there ever any diversity collapse to fix?

**Run:** 2026-08-13 · `code/experiments/e1e_collapse_diagnostic.py` · baseline EGGROLL, skewed
task, `N = 128` · raw output in `E1e-results.txt`.

Every mechanism in [E1](E1-results.md) assumed the population loses behavioural diversity over
training. Nobody checked. This checks.

## Result — it does not collapse

| gen | eff_rank | logdet | fitness spread | worst mode |
|---|---|---|---|---|
| 0 | 91.9 | −90.8 | 0.0124 | −3.22 |
| 50 | 73.2 | −105.5 | 0.0109 | −3.92 |
| 100 | 75.9 | −101.2 | 0.0127 | −3.81 |
| 200 | 66.8 | −103.6 | 0.0125 | −5.77 |
| 300 | 64.4 | −107.6 | 0.0093 | −4.06 |
| 375 | 65.1 | −107.7 | 0.0065 | −3.01 |

`eff_rank` = effective rank of the population's Tier-0 signature matrix — how many independent
behavioural directions 128 members span.

**It drifts from 92 to 65 and then flattens.** The population still spans ~65 independent
behavioural directions out of 128 members after 375 generations. `logdet` (the DvD volume) is
flat within noise from generation 50 onward. Fitness spread is flat at ~0.01 throughout.

Meanwhile `worst_mode` wanders between −3.0 and −5.8 with **no trend** — it is noise-dominated,
not collapsing.

## What this explains

**There was never any diversity collapse on this task.** The population is behaviourally diverse
from start to finish. So:

- Every mechanism in E1 was solving a problem that did not exist. That single fact explains the
  entire result set at once — why nothing beat a tuned baseline, why a random bonus matched a
  novelty bonus, and why the only things that helped were step-size knobs.
- The rare mode is lost (when it is lost) because the **update over-commits** to the dominant
  mode, not because the population stopped exploring. That is a step-size problem, and
  [E1c](E1c-results.md) confirms it is fixed by step-size tools.

This is the difference between *population* diversity and *update* diversity. ES resamples its
population from scratch every generation, so it cannot collapse the way a persistent GA
population does — the noise distribution is fixed and isotropic. What can collapse is the
**update direction**, and no amount of reweighting a diverse population changes the fact that
70% of the data points the same way.

That distinction is the most useful thing this experiment produced, and it was not in any of the
three proposals.

## Consequence for the research programme

A diversity mechanism is only worth building when the detector fires. The detector is now free
and validated:

```
eff_rank(Tier-0 signature)  and  log det(K)   logged every generation
```

If those stay flat, adding diversity machinery is guaranteed waste. This reframes the Tier-0
signature from "the metric we route into the update" (which failed) to **"the instrument that
tells us whether a mechanism is warranted at all"** (which works, and is what E0 actually
validated).

## Caveat

One task, one seed for the trace, one architecture. The claim "ES populations resample and so
cannot collapse like a persistent GA" is structural and should generalise; the claim "this
particular task never needs diversity" is specific. Finding a task where the detector *does* fire
is now the prerequisite for any further work on P1 — see
[decisions/0002](../../claude/decisions/0002-diversity-mechanisms-not-yet-warranted.md).
