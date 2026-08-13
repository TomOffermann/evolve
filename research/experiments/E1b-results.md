# E1b — The control that overturned E1's winner

**Run:** 2026-08-12 · `code/experiments/e1b_novelty_sweep.py` · 6 seeds · raw output in
`E1b-results.txt`.

[E1](E1-results.md) picked `novelty` (fixed-λ k-NN bonus on the Tier-0 signature). E1b added the
control that makes that claim falsifiable: **a random bonus at matched λ.**

## Result — the control wins

Skewed task `[.70, .20, .07, .03]`, Δ versus baseline:

| arm | Δ mean | Δ worst |
|---|---|---|
| novelty λ=0.1 | +0.032 | +0.192 |
| novelty λ=0.2 | +0.047 | +0.222 |
| novelty λ=0.3 | +0.042 | +0.087 |
| novelty λ=0.5 | +0.025 | +0.363 |
| novelty λ=0.7 | −0.145 | −1.318 |
| **RANDOM-bonus λ=0.3** | **+0.083** | **+0.733** |
| RANDOM-bonus λ=0.5 | +0.045 | +0.334 |

**A random bonus beats every novelty setting**, on both metrics. The novelty *signal*
contributes nothing measurable. E1's headline was real in the sense that the arm beat baseline —
and wrong in its explanation of why.

What survived from E1 is that novelty λ=0.7 is clearly bad, so the bonus can be overdone. What
did not survive is the reason to compute it from a diversity signature at all.

## Uniform task — the neutrality check

| arm | Δ mean | Δ worst |
|---|---|---|
| novelty λ=0.3 | +0.030 | −0.013 |
| RANDOM-bonus λ=0.3 | +0.020 | −0.003 |

Both help slightly even where there is *nothing for diversity to buy* — no rare mode, no local
optimum. A genuine diversity mechanism should be roughly neutral here. Both being mildly
positive is the tell: **whatever the bonus does, it is not diversity-specific.**

## The implied explanation

Mixing `rank(F)` with any second ranking decorrelates the update from fitness. That is
**reduced selection pressure** — it slows commitment to the dominant mode. Under that reading
the mechanism is not exploring intelligently; it is just committing less, and any noise source
does it equally well.

If true, plain hyperparameters (lower learning rate, larger σ) should reproduce the entire
effect, and no diversity mechanism is warranted on this task. → tested in
[E1c](E1c-results.md).

## Process note — a bug this experiment exposed

E1b's baseline came out at −0.6049 where E1's was −0.5961, on identical seeds and settings. The
cause: batch selection used the **global** torch RNG, so every arm's data ordering depended on
which arms had run before it. Arm execution order was an uncontrolled variable across the whole
E1 table.

Fixed in `e1_mechanisms.py` — batching now uses a generator seeded per run, identical across
arms. The E1 conclusions were already overturned on other grounds, but the numbers in
[E1-results.md](E1-results.md) carry this flaw and should not be quoted precisely. E1c reruns
the comparisons that matter with the fix and with standard errors instead of standard
deviations.
