# Stage 1: synthetic task family with known ground truth

The code is in `stage1_evolvable.py` (single file). `Stage1_Evolvable_Colab.ipynb` contains the same code, ready to run on a Colab CPU. The proposal is `../evolvable_representations.tex` (RQ1–RQ3, §6.1).

## Run

```bash
pip install torch cma scikit-learn scipy matplotlib
python stage1_evolvable.py --sanity-only                      # correctness checks (~20 s)
python stage1_evolvable.py --mode smoke --seeds 0             # pipeline test (~1 min)
python stage1_evolvable.py --mode quick --seeds 0 1           # ~17 min per seed on 2 CPU cores
python stage1_evolvable.py --mode full  --seeds 0 1 2 3 4     # + ablations eil_margin, eil_co
python stage1_evolvable.py --mode quick --set train_frac=0.1 lam_co=1.0 --out results_tf01
```

On Euler, run one job per seed (`--seeds k --out results_k`); results.json merges easily. Every run writes `log.txt`, `results.json` (per task, per run), `summary.csv`, `summary.txt` and three plots.

## What is tested

| part | content |
|---|---|
| World | 16 latent bits → `tanh` mixing → 64-d observations. Tasks: `sign(Σ_{j∈S} a_j(2s_j−1))`, \|S\|=3, giving 4480 tasks (70% meta-train, 30% held out). Related tasks change one factor or one sign. |
| Read-out | Ternary genome `g ∈ {−1,0,1}^64` voting over binary features, `sign(<g,f> − 0.5)`. Reward is batch accuracy (512 fixed samples = common random numbers); test is 4096 held-out samples. |
| Representations | `ideal` (true factors, oracle), `random` (untrained, floor), `autoencoder`, `multitask` (all 3136 heads per step, strong baseline), `multitask_co` (+ co-activation penalty), `eil` (evolution in the loop: inner (1+1) EA per task, gradient at the genome it found), `eil_full` (+ margin γ=4.5, co-activation, band penalty). Full mode adds `eil_margin` and `eil_co`. |
| Adaptation | (1+1) EA and (1+16) EA on the genome; CMA-ES on a continuous head over the same features. Tested on fresh held-out tasks (start from zero) and on related transitions (start from the pruned previous solution). |
| Measurements | Relative epistasis ε along the EA trajectory; essential task distance k (neutral drift pruned); co-activation; band mass; exact lemma check (`lemmaV` must be 0); RQ1 Spearman correlations. |

**Sanity checks** (run first, every time):

- vectorised fitness equals a naive loop;
- labels are balanced;
- the analytic ideal genome gives accuracy 1;
- the mutation operator behaves as specified;
- the straight-through forward pass equals the hard bits;
- the epistasis lemma holds exactly (0 violations);
- the (1+1) EA on OneMax-like ternary functions matches the 2enH_k bound (ratio 0.93–0.98, so the bound is nearly tight);
- the EA solves tasks on ideal features.

## Preliminary result: quick mode, one seed (pipeline check, not evidence)

`prelim_quick_seed0_summary.txt`. Key numbers are median reward evaluations to reach 0.90 with the (1+1) EA:

| rep | capacity (labels) | fresh: evals / test acc | transition: evals | ε |
|---|---|---|---|---|
| ideal | 1.000 | 134 / 1.000 | 103 | 0.57 |
| multitask | 1.000 | 172 / 0.980 | 100 | 0.63 |
| eil | 0.948 | 717 / 0.881 | 520 | 0.65 |
| **eil_full** | **1.000** | **134 / 0.992** | **112** | 0.64 |
| autoencoder | 0.974 | >3000 / 0.823 | >3000 | 0.66 |
| multitask_co | 0.999 | 616 / 0.874 | 333 | **0.30** |

What this does and does not say:

1. **eil_full reaches oracle speed on fresh tasks** and beats plain EIL by a wide margin. But the strong multi-task baseline is nearly as good. On this world, supervised training on 3136 tasks already recovers the true factors, so there is little room left to shape.
   **Next:** make generic training fail to find evolvable features. Options: few training tasks (`--set train_frac=0.05`), transitions that use factors rare in training, or noisier or more entangled observations.
2. **ε is not the whole story.** multitask_co has half the epistasis but is slower, because sparse features make the ternary vote less expressive. ε correlates with speed (ρ ≈ −0.3 to −0.4 against AUC), but capacity of the *ternary* read-out matters too.
3. **k stays large (~25) even for the oracle.** EA solutions are dense (many redundant votes), so transitions are far from the small-k regime of the theory.
   **Next:** add a parsimony tie-break (prefer fewer non-zeros on equal reward) to see whether k and transition cost drop.
4. The **ablations in full mode** (`eil_margin` vs `eil_co`) show which part of eil_full does the work. The co-activation and band penalties barely moved co-activation at λ=0.3, so the margin is the prime suspect.

Before believing any of this: ≥5 seeds, and check that controls vary (rule 4 in `claude/context.md`).
