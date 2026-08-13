# P1 — A continuous diversity score for parameter deltas

**Status:** Open. This is the core problem.

## Statement

[DEGA](../literature/dega-dipec.md) needs a distance between individuals. Its individuals are
bitstrings, its distance is Hamming. [EGGROLL](../literature/eggroll.md)'s individuals are
low-rank Gaussian perturbations `E_i = (1/√r) A_i B_iᵀ` of a `d ≈ 10^8–10^10` parameter model.
What is the right distance?

## Why the obvious answer is dead

Take two independent Gaussian perturbations in `R^d`. Their cosine similarity is distributed
roughly `N(0, 1/d)`. At `d = 10^9` that is a standard deviation of ~`3·10^-5`.

**Every pair in the population is orthogonal to every other pair, to five decimal places.**

This is not a small effect to be corrected — it is total. The pairwise distance matrix of a
freshly sampled ES population is, to measurement precision, a constant off-diagonal matrix. It
carries no ranking information. Fitness sharing, crowding, clearing, k-NN novelty — every classical
niching method built on genotype distance degenerates to a no-op. (The niching literature knows
this as the curse of dimensionality; at `d = 10^9` it is not a degradation, it is a wall.)

The deeper point: even if it *did* discriminate, it would be measuring the wrong thing. We don't
care whether two deltas are different vectors. We care whether they make the model **behave**
differently. Two perturbations that are near-orthogonal in parameter space can be functionally
identical (neuron permutation, scale symmetry under normalisation layers, null directions).
Genotype distance is not the quantity of interest; it was only ever a cheap proxy, and in
bitstring-land it happened to be a good one.

So: measure in **function space**, or measure in a **low-dimensional meaningful subspace** of
parameter space. Not in raw `R^d`.

## Hard constraints from the substrate

Any candidate must survive all five:

| # | Constraint | Consequence |
|---|---|---|
| C1 | Members are *seeds*; `A_i, B_i` are regenerated, never stored | Metric must be computable from the low-rank factors or from rollout by-products. No storing `N` dense deltas. |
| C2 | `N` ranges 10^3 … 10^6 | `O(N²)` pairwise is out above `N ≈ 10^3`. Need `O(N·k)` sketches, or a determinant on a `k`-dim embedding. |
| C3 | Update is a fitness-weighted sum of `A_i B_iᵀ` | Diversity can only enter at three points — see below. |
| C4 | `(x B_i) A_iᵀ` is already computed in the forward pass | A rank-`r` per-member functional signature is *free*. Use it. |
| C5 | Per-member fitness `F_i` (and often per-example fitness) is already computed | Anything built on `F` alone costs nothing. |

## The three injection points

Worth naming explicitly, because most of the literature conflates them and they have very
different costs and failure modes.

1. **Fitness shaping** — modify `F̃_i` with a novelty/diversity bonus.
   *NS-ES, NSR-ES, NSRA-ES, DvD.* Easy to bolt on. Biases the gradient estimate; needs a
   weight schedule, and the weight schedule is where these methods usually die.
2. **Sampling / proposal** — change the distribution the `E_i` are drawn from so the population
   is diverse *by construction*.
   *Orthogonal MC, DPP sampling, antithetic pairs, partitioned masks.* No fitness bias if done
   right, and it can strictly **reduce estimator variance** rather than trading against it.
   Cheapest place to win.
3. **Selection / archive** — keep an archive, choose which members survive or how they're weighted.
   *MAP-Elites, CMA-ME/CMA-MAE, DEGA's max-Hamming pair, DPP MAP selection.* Where the
   real quality-diversity behaviour lives, but needs a descriptor space, and a descriptor space
   is exactly what we don't have yet.

DEGA uses (3) in its diversity phase and something like (1)+(3) in its exploitation phase.

## What "good" looks like

A diversity score `D` is worth having if:

- **It discriminates.** `D` between two members separates measurably better than chance. If
  the histogram of `D` over random pairs is a spike, the metric is dead.
- **It predicts.** Members that are far apart under `D` should have *decorrelated* fitness —
  they should succeed and fail on different inputs. This is the operational definition and the
  one we should test against.
- **It's invariant to the things we don't care about.** Perturbation magnitude `σ`, neuron
  permutation, layer-wise scale.
- **It's affordable.** `O(N·k)` with `k` in the tens, no extra forward passes, or the extra
  forward passes are shared across the population.

## Next step

[E0](../experiments/E0-does-parameter-distance-mean-anything.md) — verify the concentration
argument empirically before building anything on it. One afternoon.
