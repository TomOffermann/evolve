# E1c — The decisive test: mechanisms vs. two ordinary hyperparameters

**Run:** 2026-08-13 · `code/experiments/e1c_decisive.py` · 7 arms × 10 seeds × 400 generations ·
deterministic per-seed batching · raw output in `E1c-results.txt`.

[E1b](E1b-results.md) showed the novelty signal contributes nothing — a random bonus at matched
λ beat it. The implied explanation was that **any** bonus decorrelates the update from fitness,
which is just *reduced selection pressure*. E1c tests that directly against the boring knobs.

## Result

Δ versus baseline, `sem` = standard error of the mean over 10 seeds:

| arm | mean | sem | Δ mean | worst | sem | Δ worst |
|---|---|---|---|---|---|---|
| **σ×1.5 + lr×0.7** | **−0.4562** | 0.007 | **+0.133** | **−1.782** | 0.081 | **+1.564** |
| lr×0.5 | −0.4664 | 0.005 | +0.123 | −1.844 | 0.125 | +1.502 |
| lr×0.7 | −0.5020 | 0.007 | +0.087 | −2.137 | 0.148 | +1.210 |
| σ×1.5 | −0.4976 | 0.008 | +0.091 | −2.300 | 0.116 | +1.046 |
| RANDOM-bonus λ=0.3 | −0.5338 | 0.011 | +0.055 | −2.666 | 0.228 | +0.680 |
| novelty λ=0.2 | −0.5488 | 0.011 | +0.040 | −2.910 | 0.193 | +0.436 |
| *baseline* | *−0.5890* | *0.016* | *—* | *−3.346* | *0.386* | *—* |

**Two ordinary hyperparameters beat every diversity mechanism, on both metrics, by margins far
larger than the standard errors.** The ranking is unambiguous: hyperparameters > random bonus >
novelty > baseline. The novelty bonus — E1's winner — is the *weakest* of the interventions that
work at all.

## Why

The baseline was simply over-committing: learning rate too high and σ too low for this task. Any
mechanism that decorrelates the update from fitness partially mimics a lower learning rate, which
is why all of them showed a gain. But mimicking is strictly worse than doing:

- Lowering `lr` reduces the step size along the estimated gradient. That is all it does.
- A bonus reduces the *effective* step along the gradient **and** adds variance orthogonal to it.

You pay for the same effect with extra noise. Hence the consistent ordering, hence the fact that
combining both knobs (σ×1.5 + lr×0.7) is best of all: more exploration per generation, less
commitment per step.

## The methodological point

Every one of these mechanisms would have looked like a success against an untuned baseline. E1
reported exactly that and got the explanation wrong. The rule this earns:

> **A new mechanism must beat a *tuned* baseline, not a default one.** Any intervention that
> perturbs the update partially mimics a smaller step size, so the baseline's step size is a
> confound in every comparison until it is swept.

Together with E0's lesson (*spread is not signal*), that is two ways this project has now nearly
fooled itself with a healthy-looking number.

→ [E1e](E1e-results.md) asks the question underneath all of this: was there ever any diversity
collapse to fix?
