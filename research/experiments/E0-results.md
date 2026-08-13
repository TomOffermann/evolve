# E0 — Results

**Run:** 2026-08-12 · `code/experiments/e0_metric_sanity.py` · raw output in
`E0-results-small.txt` / `E0-results-big.txt`.

Setup: synthetic 4-mode grammar, char-level MLP LM, rank-1 EGGROLL, `N = 256`, `σ = 0.02`.
Two scales: **8,144** and **81,328** parameters. Metrics computed on probe batch P
(648 examples), target measured on **held-out** batch Q (648 examples).

`ρ` = Spearman rank correlation between a metric's pairwise distance and behavioural
decorrelation on held-out data. Read against **RANDOM** (floor) and **CEILING** (the same
behavioural statistic computed on P — how well behaviour transfers between batches at all).

## Small model (8,144 params)

| metric | gen 0 | gen 40 | gen 120 | gen 300 |
|---|---|---|---|---|
| RANDOM (floor) | 0.000 | −0.010 | −0.003 | 0.006 |
| 1 param cosine | 0.098 | 0.063 | 0.072 | 0.076 |
| 4 signed Hamming on delta | 0.015 | 0.019 | 0.024 | 0.018 |
| 5 per-layer energy profile | −0.002 | 0.001 | 0.004 | −0.000 |
| 6 active subspace (**Prop C**) | — | 0.019 | 0.013 | 0.019 |
| A1 own-layer delta, sketched | 0.070 | 0.047 | 0.043 | 0.038 |
| A2 total output delta, sketched | 0.157 | 0.086 | 0.082 | 0.087 |
| **A3 total delta, task-projected** | **0.838** | 0.333 | 0.238 | 0.202 |
| **A0 per-example fitness (Prop A T0)** | **0.872** | **0.857** | **0.786** | **0.760** |
| CEILING | 0.872 | 0.857 | 0.786 | 0.760 |

## Verdicts

### H1 — concentration. **Confirmed, to three significant figures.**

| params `d` | predicted cos std `1/√d` | measured |
|---|---|---|
| 8,144 | 0.0111 | 0.0109 – 0.0112 |
| 81,328 | 0.0035 | 0.0035 |

The theory in [P1](../problems/P1-continuous-diversity.md) is exactly right, and the effect
**worsens with scale** as predicted (param-cosine ρ falls from 0.076 → 0.024 when the model
grows 10×). Extrapolating to `d = 10^9` gives a spread of `3·10⁻⁵`. Settled.

### H2 — parameter-space metrics carry no behavioural signal. **Confirmed.**

Everything in genotype space sits at 0.00–0.10 against a ceiling of 0.76–0.87. Dead.

### H3 — function-space metrics work. **Confirmed for Tier 0 only. Tier 1 falsified.**

This is where the experiment disagreed with what I had written, in two places.

## Finding 1 — the "free lunch" signature does not work

[Proposal A](../proposals/A-functional-signature-diversity.md) claimed EGGROLL's
already-computed activation delta `(xB_i)A_iᵀ` gives a usable functional fingerprint at zero
cost (constraint C4). **It does not:** ρ = 0.04–0.07, barely above floor. Two reasons, both
now understood:

- It captures only the *output layer's own* low-rank contribution, not the total functional
  change — perturbations to earlier layers propagate through and are missed. Using the total
  output delta instead roughly doubles ρ (A1 → A2), but 0.09 is still near the floor.
- More importantly, see Finding 2.

## Finding 2 — the signature must be task-projected, not sketched *(new, not in the write-up)*

Same delta, two readings:

| reading | ρ (gen 0) |
|---|---|
| random JL sketch of the output delta (A2) | 0.157 |
| **the same delta read along the objective direction (A3)** | **0.838** |

A 5× jump from a projection change. The reason: **the output space is mostly task-irrelevant
variation, and a JL sketch preserves it faithfully — which is exactly the problem.** Random
projection is distance-preserving, and here that is a defect, not a feature. The objective
already specifies which direction in output space matters; use it.

Corollary that decides the design: A3 (a *linear* proxy for the objective) decays over
training, 0.84 → 0.20, as the network saturates and the logit delta decouples from the actual
likelihood change. **A0 — just the per-example objective value — stays at 0.76+ throughout.**

So Proposal A collapses to its cheapest tier. The free thing wins, the clever things lose, and
we now know why. Tier 0 is also trivially objective-agnostic: any objective with per-example
structure has it already.

## Finding 3 — Proposal C's premise is false here

[Proposal C](../proposals/C-active-subspace-diversity.md) argued that projecting onto the
optimiser's active subspace restores meaning to parameter distance. Measured: **ρ = 0.010–0.019
at both scales.** Not a scale artifact, not a warm-up artifact.

The instructive part is *how* it fails. C's spread is **0.20 — eighteen times wider than raw
param cosine's 0.011.** It discriminates beautifully. It just discriminates on nothing.

> **Spread is not signal.** A metric that varies a lot and means nothing is more dangerous
> than one that doesn't vary, because it looks like it is working.

Stated precisely, C's premise requires that a perturbation's *behavioural* effect be dominated
by its active-subspace component. E0 says it is not: the perturbations are isotropic, so their
projections onto any fixed `k`-dim basis are just random `k`-dim vectors.

**C is not dead as a variance-reduction idea** — the orthogonal-sampling MSE result is a
theorem and stands untouched, and it was never tested here. It is dead as a *diversity metric*.
Those were two separate claims in the proposal and only one of them fell.

## What this changes

- **Build order was C → A → B. It is now A(Tier 0) → B → C(variance reduction only).**
  Tier 0 is ~20 lines and already validated; it becomes the measurement infrastructure for
  everything else.
- [Proposal B](../proposals/B-modular-partition-diversity.md) is untouched by E0 — it was
  never tested, because it needs partitioned perturbation that doesn't exist yet. It is now
  the only substantive open bet, and E0 gives it a validated yardstick to be judged against.
- Metric design rule to carry forward: **project along the objective, don't sketch the outputs.**

## Honest limits

One task, one architecture, two scales, rank-1, one `σ`, `N = 256`. The concentration result
(H1) is architecture-independent — it is arithmetic, and it matched to three digits. The metric
*rankings* are not established beyond this setting. In particular A3's decay may be specific to
the `tanh` saturation in this model, and C deserves one more test on a model large enough for
the active subspace to be genuinely low-dimensional relative to `d`.
