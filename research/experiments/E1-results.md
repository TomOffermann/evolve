# E1 — Do any diversity mechanisms actually improve optimisation?

**Run:** 2026-08-12 · `code/experiments/e1_mechanisms.py` · 8 arms × 6 seeds × 400 generations.

> ⚠️ **Superseded in its conclusion by [E1b](E1b-results.md), and its numbers carry a bug.**
>
> - E1's winner (`novelty`) was beaten by a **random bonus at matched λ**. The diversity signal
>   contributes nothing; the gain came from decoupling the update from fitness.
> - Batch selection used the global torch RNG, so every arm's data ordering depended on which
>   arms ran before it. Fixed; **do not quote these numbers precisely.**
>
> What still stands: the analysis of *why the losers lost*. Those arms failed by margins far
> larger than the ordering effect, and the failure mechanisms are the durable content of this
> page.

[E0](E0-results.md) showed the Tier-0 signature *predicts behaviour*. Necessary, not sufficient.
E1 asks the question that decides whether the direction is worth anything: routed into the
update, does it make the optimiser **better**?

## Design

Task: grammar with **skewed mode frequencies** `[.70, .20, .07, .03]`. A greedy optimiser fits
the dominant mode and abandons the rare ones — the same failure as diversity collapse in RLVR.
This choice is load-bearing: *on a uniform task with no local optima, nothing can help and every
arm ties*, which would tell us nothing.

Every arm gets the same evaluation budget (`N = 128` × 400 generations). All mechanisms consume
the same free Tier-0 signature and return a per-member weight vector — drop-in replacements for
`rank_normalise(F)`. Two numbers, and they are not the same question:

- **mean** — held-out log-likelihood. Raw performance.
- **worst** — held-out log-likelihood on the *rarest* mode (3% of data). What diversity is for.

## Results

| mechanism | mean | ± | Δ | worst | ± | Δ |
|---|---|---|---|---|---|---|
| **novelty** (k-NN bonus, λ=0.3) | **−0.5294** | 0.035 | **+0.067** | **−2.633** | 0.68 | **+1.008** |
| dpp_marginal (DvD volume) | −0.5371 | 0.040 | +0.059 | −2.670 | 0.78 | +0.970 |
| *baseline (vanilla EGGROLL)* | *−0.5961* | *0.025* | *—* | *−3.640* | *0.82* | *—* |
| dega_phases | −0.6726 | 0.039 | −0.077 | −4.053 | 0.77 | −0.413 |
| antithetic | −0.7754 | 0.064 | −0.179 | −5.610 | 1.50 | −1.970 |
| whitened (GLS) | −0.8954 | 0.051 | −0.299 | −6.104 | 0.82 | −2.464 |
| novelty_adapt (NSRA-ES) | −1.1744 | 0.052 | −0.578 | −7.418 | 0.87 | −3.778 |
| dipec (subsampled step) | −2.3939 | 0.241 | −1.798 | −17.283 | 2.30 | −13.643 |

Five of eight arms are **worse than doing nothing**. That is the useful part of this table.

## Why the losers lost

**`dipec` — catastrophic (−1.80).** DEGA's subsampled improving step does not survive the move
to ES, and the reason is structural. In DEGA the improving mask has *meaningful components*:
dropping some of them keeps a valid, smaller improvement. In ES the population is a Monte-Carlo
gradient estimate, and randomly discarding 3/4 of it simply multiplies the estimator variance by
four. Same operator, opposite effect, because the object being subsampled is a different kind of
thing. **This is the clearest evidence so far that DEGA mechanisms do not port for free.**

**`antithetic` — worse (−0.18), and this one is genuinely surprising.** Mirrored pairs are a
textbook ES control variate. But at fixed budget they buy variance reduction by halving the
number of *independent* exploration directions, from N to N/2. In high dimension the count of
independent directions dominates. This is a plausible reason EGGROLL samples independently
despite the control variate being free — a note in `literature/eggroll.md` flagged the omission
as "a free win they left on the table". It is not. Corrected there.

**`whitened` — worse (−0.30).** My own idea, from Proposal C's surviving thread: treat the ES
update as an average over *correlated* observations and apply GLS, `w ← (K + ridge·I)⁻¹ w`. The
reasoning was that redundant members double-count the same evidence. Measured: it makes things
substantially worse. Inverting a noisy 128×128 correlation matrix amplifies exactly the
directions the signature estimates least reliably. Diversity-as-variance-reduction does not work
here; diversity-as-exploration does.

**`novelty_adapt` — much worse (−0.58) than fixed-λ novelty.** NSRA-ES's schedule ratchets λ up
during stagnation and recovers too slowly, so it spends most of training at λ ≈ 0.8 — optimising
for novelty over fitness. Proposal A predicted this failure in writing ("the weight schedule is
where these methods usually die") and then it happened to the adaptive arm. The fixed schedule
is not just simpler, it is better.

**`dega_phases` — slightly worse (−0.077) and 2.5× slower.** The phase structure adds machinery
and a threshold hyperparameter for no measured gain over just applying the bonus continuously.

## What won

**`novelty` — a fixed-λ k-NN novelty bonus on the Tier-0 signature.** Best on both metrics, and
the simplest of all the diversity arms: no state, no adaptation, no matrix inverse, no phase
controller. `dpp_marginal` matches it within noise while costing an N×N inverse — the fancier
aggregator buys nothing here.

The `worst` gain (+1.0 nats on the rarest mode) is the mechanism doing exactly what it was
designed to do: keeping members alive that are good at the 3% of the data everyone else abandons.

## Caveat that E1 cannot settle on its own

λ = 0.3 and k = 10 were arbitrary, and a bonus that partly decouples the update from fitness
injects exploration *whatever* it is computed from. Without a control, "novelty helps" is
unfalsifiable. → [E1b](E1b-results.md) adds a **random-bonus arm** at matched λ, a λ sweep, and
a uniform-task run where the mechanism should be *neutral*.

**No commitment until E1b returns.**
