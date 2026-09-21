# Stage 1, round 1: harder world, ablation, stronger regularisers (2 seeds each)

Raw: `results_next/` (summary.txt per run). Sanity checks: all passed. Lemma violations: 0.

All statistics are paired Wilcoxon tests over 48 tasks (24 × 2 seeds). "evals" means the median reward evaluations for the (1+1) EA to reach 0.90.

## Decisions (rules were fixed before the run, see proposal §7.6)

| run | rule | outcome | verdict |
|---|---|---|---|
| **Harder world** (`train_frac=0.05`) | eil_full needs fewer evals than multitask, p < 0.01, at similar accuracy | **fresh:** 150 vs 202 evals (0.74×), **p = 0.0087**; AUC p = 5e-5; test 0.994 vs 0.981. **transitions:** 98 vs 107, p = 0.41 (AUC p = 2e-6) | **Supported on fresh tasks, weakly** (2 seeds, just under the threshold). Transitions: n.s. on evals, significant on AUC. |
| **Ablation** | eil_margin ≈ eil_full and eil_co ≈ eil | eil_margin vs eil_full: p = 0.32 / 0.61. eil_co vs eil: p = 0.09 / 0.50. eil_margin vs eil: 0.30× / 0.17× evals, p < 1e-12 | **Confirmed:** the margin term does the work; the co-activation penalty does nothing useful. |
| **λ = 1** | ε drops ≥ 30 % with accuracy within 1 point and no slow-down | ε 0.625 → 0.547 (−12 %); 1.41× slower (p = 7e-6); test 0.986 → 0.943 | **Fails** in the predicted way: this confirms the capacity axis. |

## What this means

1. **First separation between shaped and generic representations.** With about 224 training tasks, both EIL variants beat multi-task training on fresh held-out tasks, and they also reach *higher* final accuracy. Notably, multi-task features are **equally informative** (factor decodability 1.000, logistic capacity 1.000), yet less usable by the EA. That is exactly the claim: *good features for a gradient head ≠ good features for an evolutionary head* (the ANIL contrast).
2. **Why does plain EIL work at `train_frac=0.05` but not at 0.7?** With fewer tasks, each task's archived genome gets about 14× more inner-EA iterations (about 860 vs 60 visits), so gradients are taken at *well-optimised* genomes. At 0.7 the margin term compensates. Hypothesis: EIL needs gradients at near-optimal genomes, and the margin makes it robust to under-optimised ones. This is testable with an `inner_iters` sweep at `train_frac=0.7`.
3. **Relative epistasis ε is not a valid predictor.** Its correlation with speed *flips sign* depending on what varies: −0.30 in round 0, but +0.32 (ablation) and +0.73 (λ = 1) against AUC. Regularisers lower ε by making features sparser, which also removes improving moves. So ε measures "how interacting" the landscape is, not "how hard". The runtime-relevant quantities are the first-order signal (E|Δ₁|) and the probability that a single mutation improves. Both are now recorded (code updated, smoke-tested), and RQ1 is re-tested on them.
4. **The margin regime of the lemma is what helps, not the sparse regime.** A clean, theory-relevant result: pushing examples out of the |m| < 4 band via redundancy helps; reducing co-activation hurts capacity.

## Caveats

- 2 seeds, and the fresh-task p-value is borderline.
- **Rule 2 (tuned baseline):** multitask was not re-tuned for 224 tasks, and EIL uses about 4× more wall-clock (80 s vs 20 s).
- **Round 3** addresses both: 5 seeds, plus a compute-matched multitask (12 000 steps, head lr 3e-3).

## Next runs

- Round 3 (notebook §7).
- Round 2 (notebook §6: parsimony, weighted tasks, size split).
- An `inner_iters` sweep at `train_frac=0.7`.
