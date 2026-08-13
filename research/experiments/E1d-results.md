# E1d — The third injection point: selection

**Run:** 2026-08-13 · `code/experiments/e1d_niche.py` · 7 arms × 10 seeds × 400 generations ·
raw output in `E1d-results.txt`.

[P1](../problems/P1-continuous-diversity.md) names three injection points. E1 tested **fitness
shaping** (novelty, DvD, random) and **sampling** (antithetic). **Selection** — where
quality-diversity actually lives — was never tried, and it had the best a-priori argument:

> On a skewed task the failure is not that rare-mode specialists are *absent*. It is that they
> are **outvoted**. If 3% of the data is mode D, members relatively good at D are a small
> minority and their signal is swamped in the fitness mean.

Niche balancing addresses that directly: k-means the Tier-0 signatures, then scale each member's
vote by `1/√(niche size)` (soft) or equalise total weight per niche (hard).

## Result

| arm | mean | sem | Δ mean | worst | sem | Δ worst |
|---|---|---|---|---|---|---|
| niche k=8 **+ tuned HP** | −0.4525 | 0.006 | +0.137 | −1.748 | 0.112 | +1.598 |
| **tuned HP alone** (σ×1.5, lr×0.7) | −0.4562 | 0.007 | +0.133 | −1.782 | 0.081 | +1.564 |
| RANDOM-bonus λ=0.3 | −0.5338 | 0.011 | +0.055 | −2.666 | 0.228 | +0.680 |
| niche soft k=8 | −0.5775 | 0.010 | +0.012 | −3.269 | 0.192 | +0.077 |
| niche hard k=8 | −0.5855 | 0.016 | +0.004 | −3.296 | 0.343 | +0.050 |
| niche soft k=16 | −0.5873 | 0.013 | +0.002 | −3.342 | 0.250 | +0.004 |
| *baseline* | *−0.5890* | *0.016* | *—* | *−3.346* | *0.386* | *—* |

**Niching on its own is indistinguishable from baseline** — every Δ is smaller than its own
standard error. It does not even reach the random bonus.

**Niching on top of tuned hyperparameters adds +0.004 mean / +0.034 worst** over tuned
hyperparameters alone, against standard errors of 0.006 and 0.112. That is zero.

## What it means

All three injection points have now been tested and all three fail:

| injection point | mechanisms tried | outcome |
|---|---|---|
| fitness shaping | novelty, adaptive novelty, DvD marginal, random control | beaten by σ/lr; signal contributes nothing beyond noise |
| sampling | antithetic pairs | worse than independent sampling |
| **selection** | niche balancing (soft/hard, k = 4/8/16) | indistinguishable from baseline |

The best a-priori argument in the whole programme — that rare-behaviour members are outvoted —
turns out to be *true but irrelevant*. Rebalancing their votes changes nothing, because
[E1e](E1e-results.md) shows the population is behaviourally diverse the whole time anyway. You
cannot fix a representation problem that does not exist.

This closes the mechanism search. See
[decisions/0002](../../claude/decisions/0002-diversity-mechanisms-not-yet-warranted.md).

## One thing worth keeping

`niche_balanced(hard=True)` at 40 generations was *catastrophically* worse than baseline
(−1.28 vs −0.59) but roughly neutral by generation 150–400. Aggressive reweighting mostly costs
early convergence speed. If a mechanism is ever warranted, that argues for annealing it in rather
than applying it from step zero — the same shape as CMA-MAE's annealed archive threshold.
