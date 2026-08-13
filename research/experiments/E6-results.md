# E6 — Proposal B: partitioned perturbation

**Run:** 2026-08-13 · `code/experiments/e6_partitioned.py` · countdown post-training, 250
generations, 3 seeds · raw output in `E6-results.txt`.

The first test of [Proposal B](../proposals/B-modular-partition-diversity.md) — the one idea in
the programme never tried. Every mechanism E0–E4 killed was a population-*reweighting* scheme;
this one changes **what gets perturbed**, so none of those negative results apply to it.

Parts here are the three weight matrices: `emb`, `h`, `out`.

## Results

| arm | pass@1 | sem | Δ | pass@16 | sem | Δ |
|---|---|---|---|---|---|---|
| global (iid) | 0.1211 | 0.005 | — | 0.1406 | 0.008 | — |
| **partitioned n=1** | 0.1107 | 0.015 | −0.010 | **0.2344** | 0.031 | **+0.094** |
| partitioned n=2 | 0.1003 | 0.017 | −0.021 | 0.2109 | 0.028 | +0.070 |
| interleaved | 0.1185 | 0.013 | −0.003 | 0.1797 | 0.006 | +0.039 |

SFT base: pass@1 0.0547, pass@16 0.2109.

**partitioned n=1 gains +0.094 pass@16 against a combined sem of 0.032 — +2.9 sem — while
costing 0.010 pass@1, which is 0.65 sem, i.e. nothing.** It also lands *above* the SFT base on
pass@16 (0.2344 vs 0.2109), meaning it improved pass@1 without paying for it in diversity at all.

Fewer active parts is better (n=1 > n=2 > interleaved > global), which is at least the ordering
Proposal B predicts: more structural separation, more preserved diversity.

## The per-part utility trace

The interpretability artefact that global ES structurally cannot produce (normalised share):

| arm | emb | h | out |
|---|---|---|---|
| partitioned n=1 | 0.26 | 0.33 | 0.41 |
| partitioned n=2 | 0.32 | 0.33 | 0.35 |
| interleaved | 0.30 | 0.33 | 0.36 |

The output layer carries the most learnable signal and the embedding the least, which is
plausible for a task where reward depends on the final token distribution. With n=1 the contrast
is sharpest — as it should be, since n=2 blurs credit across two parts per member.

This is P2's credit-assignment claim working: a 3-dimensional signal you can read, instead of a
10⁵-dimensional one you cannot.

## ⚠️ Do not call this a win yet — the control is missing

The obvious null explanation is **not** tested, and it is the same one that killed the last two
positive results in this project:

> **Partitioned perturbation is a smaller effective step.** With `n_active=1` of `P=3` parts, a
> member's perturbation has roughly 1/3 the squared norm of a global one. E1c established that a
> smaller step slows commitment to the dominant mode and preserves pass@k — and that plain
> hyperparameters reproduce the entire effect of every mechanism that appeared to work.

If a global run at matched perturbation norm reproduces +0.094 pass@16, then partitioning *per
se* does nothing and this is a step-size result wearing a structural costume.

Two further gaps:

- **σ was not logged per arm.** The resolution rule adapts σ independently in each arm, so the
  arms may not be at comparable effective steps at all, in either direction.
- **3 seeds.** The same power problem that made E5 uninterpretable. The pass@16 sems here (0.031,
  0.028) are large.

→ **E8** (`code/experiments/e8_partition_control.py`, running) runs the matched-step control and a global σ sweep at 8 seeds. Nothing
should be concluded from E6 until it returns.

## What is *not* at risk from that control

The per-part utility trace does not depend on whether the pass@16 gain survives. It is a
measurement that global ES cannot make, and it costs nothing. Even in the worst case — E8 shows
the gain is entirely step size — partitioned ES still buys credit assignment, and that was one of
P2's three claims independently of the diversity one.
