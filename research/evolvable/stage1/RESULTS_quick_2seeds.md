# Stage 1: quick mode, 2 seeds (Colab run, 2026-09-21)

Raw output: `quick_2seeds_summary.txt`. Sanity checks: all passed. Lemma violations: 0.

Numbers are median reward evaluations for the (1+1) EA to reach 0.90. Significance comes from paired Wilcoxon tests over the 48 tasks (same tasks for every representation).

| comparison | fresh tasks | transitions |
|---|---|---|
| eil_full vs multitask | ratio 1.05, p = 0.91; test 0.985 vs 0.986 | ratio 0.97, p = 0.49 |
| eil_full vs eil | ratio 0.23, p < 1e-13 | ratio 0.15, p < 1e-13 |
| eil_full vs ideal | ratio 1.19, p = 0.02 | ratio 1.29, p = 0.41 |
| multitask vs ideal | ratio 1.34, p = 0.008 | ratio 1.28, p = 0.03 |

## Findings

1. **No separation between shaped and generic representations (RQ2/RQ3 not answered yet).** eil_full and multitask are indistinguishable, and both are close to the oracle. This benchmark cannot exhibit the failure that shaping is meant to fix (rule 3): supervised training on 3136 tasks already recovers the true factors.
2. **Something inside eil_full matters a lot:** it is 4–7× faster than plain EIL. Suspect: the margin (γ = 4.5), since co-activation barely moved at λ = 0.3. The full-mode ablations eil_margin / eil_co decide this.
3. **Low epistasis is not enough.** multitask_co halves ε (0.30 vs 0.63) but adapts about 4× slower, because sparse features make the ternary vote less expressive. ε correlates with speed across all tasks (ρ = −0.30 fresh, −0.41 transitions vs AUC; p < 1e-6), but read-out capacity is a confounder the theory has to include.
4. **The discrete EA depends on the representation far more than CMA-ES does.**

   | representation | EA (fresh) | CMA (fresh) | EA (transitions) | CMA (transitions) |
   |---|---|---|---|---|
   | ideal | 144 | 158 | 84 | 132 |
   | eil_full | 170 | 172 | **96** | **192** |
   | autoencoder | 2585 | 434 | >3000 | 453 |

   On good features the ternary EA matches CMA on fresh tasks and is about 2× faster on transitions. On generic features it collapses. This supports the premise that the landscape is created by the representation.
5. **k stays ~25 even for the oracle.** EA genomes are dense, so the small-k regime of the theory is never reached.

## Next runs (in order)

1. **Harder world, where generic training should fail:** `--set train_frac=0.05` (about 220 training tasks).
2. **Ablations:** `--mode full --seeds 0 1` (eil_margin vs eil_co).
3. **Regulariser strength:** `--set lam_co=1.0 lam_band=1.0`.
4. **Code changes:** a parsimony tie-break (sparser genomes → small k), and a held-out *task type* split (train on 3-factor rules, test on 5-factor rules or on unseen factor combinations).
