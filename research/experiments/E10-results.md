# E10 — Why does partitioning work? Part 1: the mechanism confirmed

**Run:** 2026-08-13 · `code/experiments/e10_why.py` · countdown, N=128, 40 independent estimates
per sampler · raw output in `E10-results.txt`.

[E8](E8-results.md) and [E9](E9-results.md) established *that* partitioned perturbation works.
Neither explained *why*, and the first explanation I reached for was wrong — partitioning does not
make the aggregate update sparse.

## The hypothesis, and the prediction that makes it falsifiable

To first order a global member's fitness is `F(W) + σ Σ_q ⟨∇_q, E_q⟩`, so the estimator for block
*p* picks up terms `⟨∇_q, E_i^q⟩ E_i^p` from every other block. Mean-zero, but **not**
variance-zero: global ES cannot tell whether a member scored well because of its embedding
perturbation or its output perturbation.

```
global block-p variance      ∝ (Σ_q ‖∇_q‖²) / N
partitioned block-p variance ∝ P·‖∇_p‖² / N        (only N/P members inform block p)
```

**Prediction:** partitioning improves block *p* iff `‖∇_p‖² < mean_q ‖∇_q‖²`. It should help
below-average blocks and **hurt the dominant ones**. That second half is what makes this a real
test — a mechanism that just made everything better would be unfalsifiable.

## Measuring it without a reference

The obvious design — compare estimates against a large-N reference — **does not work here**, and
it is worth recording why. An N-sample ES estimate aligns with the true gradient by roughly
`√(N/d)`, which at N=128 over 94k parameters is 0.037. Comparing two such estimates gives a
product of two tiny numbers; a first attempt with N=512 as reference produced cosines of 0.003,
i.e. it measured the reference's own noise.

**Split-half reliability** avoids the problem entirely: draw many independent estimates of the
same quantity and take the mean pairwise cosine among them. Two independent estimates agree only
to the extent that both contain signal. It needs no reference and is scale-invariant, so samplers
with different effective sample counts stay comparable.

## Result — 3/3, including the counterintuitive half

| block | dim | ‖∇_p‖ | share of ‖∇‖² | vs mean | rel(global) | rel(partitioned) | Δ | sem |
|---|---|---|---|---|---|---|---|---|
| emb | 5,504 | 36.5 | 0.061 | 0.18 | 0.0017 | 0.0029 | **+0.0012** | 0.0005 |
| h | 55,296 | 112.4 | 0.579 | 1.74 | 0.0003 | 0.0001 | −0.0001 | 0.0002 |
| out | 33,024 | 88.6 | 0.360 | 1.08 | 0.0012 | 0.0006 | **−0.0005** | 0.0002 |

**Every sign matches the prediction.** Partitioning improves the estimate for the one
below-average block (+2.4 sem) and degrades it for both above-average blocks (−2.5 sem on `out`;
`h` is directionally right but within noise at −0.5 sem).

The gradient is genuinely unequal across blocks — `h` alone carries 58% of ‖∇‖² and `emb` only 6%
— which is what gives the prediction something to bite on.

## What this explains

Partitioning **reallocates estimator quality from strong blocks to weak ones.** That accounts for
both halves of the E8/E9 result at once:

- **Lower pass@1.** The dominant blocks drive raw task progress, and their estimates get worse.
  Measured cost: ~1.8 sem of pass@1. Predicted, not merely observed.
- **Preserved pass@16.** The dominant blocks are also where greedy sharpening happens. Degrading
  their estimate slows the collapse.

## The objection this raises, and why E9 already answers it

*"So partitioning is just a worse optimiser on the directions that matter, and less optimisation
means less collapse."* That is the E9 null again.

E9 rejects it **at the frontier level**: at matched pass@1, partitioned still holds more pass@16
(+0.047 at pass@1 0.110). So it is not less optimisation, it is **differently distributed**
optimisation. At equal total progress the partitioned model has taken a different route — more
movement in the weak block, less in the dominant ones — and that route costs less pass@k.

The three results now form one account: E8 (not step size), E9 (not less progress), E10 (a
reallocation of estimator quality across blocks, with the crossover exactly where the variance
inequality puts it).

## Follow-ups this opens

1. **Equalise the blocks.** If the effect comes from unequal ‖∇_p‖, it should shrink when the
   gradient is balanced across parts. A partition chosen to equalise ‖∇_p‖ is a direct test.
2. **Partition selectively.** Give `emb` dedicated members while perturbing `h` and `out`
   globally. If the account is right, that should get the pass@k benefit at less pass@1 cost —
   and it is a better operator than uniform partitioning.
3. **Learn the partition from ‖∇_p‖** — P2's "higher-level parameters", now with a concrete
   objective instead of a hand-drawn split.

Point 2 is the interesting one: the current operator is uniform because it was the simplest thing
to build, and this measurement says uniform is *not* optimal.

## Part 2 — drift after 250 generations

| block | global ‖ΔW‖ (% of base norm) | partitioned ‖ΔW‖ (% of base norm) |
|---|---|---|
| emb | 3.201 (9.8%) | 1.112 (3.4%) |
| h | 10.033 (27.0%) | 4.249 (11.4%) |
| out | 7.785 (42.9%) | 3.686 (20.3%) |
| **total** | **13.096** | **5.733** |

pass@16: 0.1458 (global) vs 0.2331 (partitioned).

**Partitioned drifts 2.3× less**, and roughly uniformly across blocks (2.1–2.9× each). That is
the *norm* half of arXiv:2601.20861's diagnosis — they blame dense **high-norm** updates for ES
forgetting. Partitioning does not address the density half (the aggregate update is dense in
both, as corrected above), but it does substantially reduce the norm.

### ⚠️ This is uncontrolled, and the confound is the familiar one

Partitioned also *progresses* less over 250 generations — pass@1 ~0.104 vs ~0.120 — and a model
that learns less naturally moves less. **Drift at equal generations is the same confound E9 had
to control for pass@16, one level down.** A crude normalisation (drift per unit pass@1 gained)
favours partitioned ~1.5×, but that is not a substitute for the controlled comparison.

→ **[E11](E11-results.md)** measures drift at **matched pass@1**, using E9's checkpoint trick so
one run yields the whole drift-vs-pass@1 curve. Ratio < 1 there means partitioning genuinely
reaches a given capability with a smaller parameter-space excursion. Ratio ≈ 1 means the drift gap
was just less progress, and only the pass@k result stands.

**Do not cite the 2.3× figure without E11.**
