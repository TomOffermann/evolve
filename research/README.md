# Research

Working notes for the evolutionary side of Evolve. Everything here is either
**Settled** (we decided / we verified) or **Open** (we're still guessing). Say which.

## The program in one paragraph

We use [EGGROLL](literature/eggroll.md)-style low-rank Evolution Strategies as the
optimizer substrate, because it is the first ES variant that is actually fast enough
on real models. On top of that substrate we want to port ideas from the GA / nature-inspired
side of the field — starting with [DEGA/DiPEC](literature/dega-dipec.md)'s
diversity-preserving exploitation. The blocker is that DEGA's diversity signal is
**Hamming distance on bitstrings**, and our genotypes are **Gaussian perturbations in
10^9 dimensions**. Finding an honest continuous replacement is problem #1.

## Layout

| Path | Contents |
|---|---|
| `literature/` | Paper notes. `README.md` is the annotated bibliography. |
| `problems/` | Problem statements. The things we're actually trying to solve. |
| `ideas/` | Wide, unfiltered idea lists. Cheap to add to, low bar. |
| `proposals/` | Worked-out methods. High bar: must have intuition, math, cost, and a way to be wrong. |
| `experiments/` | Experiment designs + results. |

Code lives in top-level `code/`. `code/experiments/<id>.py` matches `research/experiments/<id>`.

**The framework** (2026-08-13): `code/evolve/` is now a proper library — three pluggable seams
(`Sampler`, `Weighting`, `SigmaRule`), parallel by construction, with every operator carrying its
measured verdict. `python3 -m evolve.registry` prints the table. See
[decision 0004](../claude/decisions/0004-framework-architecture.md) and `code/README.md`.

## Open problems

- **[P1 — Continuous diversity](problems/P1-continuous-diversity.md)** *(core)*
  What replaces Hamming distance when the genotype is a Gaussian parameter delta?
- **[P2 — Structured / partitioned perturbation](problems/P2-structured-perturbation.md)**
  Perturb *parts* of the network instead of all of it. Forced modularity, credit assignment,
  much bigger swarms, and — usefully — a genotype that has a discrete component again.
- **[P3 — Objectives and hardware](problems/P3-objectives-and-hardware.md)**
  EGGROLL beyond RL (JEPA, self-supervised, non-differentiable objectives), and what it
  takes to run this on a MacBook / 3090 / a pile of 1080 Tis.

## Current proposals

Three worked-out candidates for P1. **[E0 has now been run](experiments/E0-results.md)** and it
reordered them:

| | Proposal | Verdict | ρ vs held-out behaviour |
|---|---|---|---|
| A | [Functional Signature Diversity](proposals/A-functional-signature-diversity.md) | Tier 0 validated as an **instrument** (E0). As a *mechanism*: **not established** — E4's +2.7 sem was measured against a control with a frozen seed; against one that varies it is +1.1/+1.7 sem ([E7](experiments/E7-results.md), [decision 0005](../claude/decisions/0005-novelty-not-established.md)) | **0.76 – 0.87** |
| B | [Modular Partition Diversity](proposals/B-modular-partition-diversity.md) | ✅ **validated** — +5.0 sem vs tuned global (E8), frontier dominates at matched pass@1 (E9). [Decision 0006](../claude/decisions/0006-partitioned-perturbation-works.md) | — |
| C | [Active-Subspace Diversity](proposals/C-active-subspace-diversity.md) | **falsified** as a metric; survives as variance reduction | 0.010 – 0.019 |

**Proposal B is adopted** for sparse-reward post-training where pass@k matters
([decision 0006](../claude/decisions/0006-partitioned-perturbation-works.md)). No *weighting*
mechanism is established ([decision 0005](../claude/decisions/0005-novelty-not-established.md)).
The pattern: every mechanism that failed reweighted the population; the one that worked changed
**what gets perturbed**. See [proposals/README.md](proposals/README.md).

## The benchmark was wrong, and rebuilding it changed the answer

**2026-08-13.** E0/E1 ran on a synthetic grammar with **dense log-likelihood** reward. EGGROLL's
actual case is RL-style post-training with **sparse verifiable** reward, and diversity collapse is
a failure of *that* regime. The old task could not exhibit the failure, so it could not test any
mechanism meant to prevent it — E1's "no mechanism helps" was drawn on a benchmark where nothing
could have.

Two realistic benchmarks now exist:

| benchmark | regime | failure it exhibits |
|---|---|---|
| [**E2** countdown post-training](experiments/E2-results.md) | sparse binary verifiable reward, 94k-param policy | **RLVR diversity collapse**: pass@1 0.055 → 0.124, pass@16 0.211 → **0.164** |
| [**E3** JEPA on pendulum](experiments/E3-results.md) | dense reward, self-supervised | representation collapse; tests non-differentiable objective terms |

What changed as a result:

- **The prerequisite from [decision 0002](../claude/decisions/0002-diversity-mechanisms-not-yet-warranted.md)
  is met** — a task that actually collapses now exists.
- The diversity signal appeared to separate from its random control (E2/E4) — **later retracted**
  when the control turned out to have a frozen RNG seed (E7, decision 0005).
- **The classical 1/5th rule inverts under sparse reward** and a *resolution rule* replaces it
  (E2 Findings 3–4) — it matches a hand-swept σ without the sweep.
- **A non-differentiable objective term beats its differentiable surrogates** at its own target
  (E3 Finding 1), which is P3's argument made concrete.
- Two implementation bugs the sparse regime exposed: tied ranks in `rank_normalise`, and sampling
  noise swamping the ES signal without common random numbers.

## What E1 settled — on a benchmark now known to be inadequate

Run 2026-08-13. Eight diversity mechanisms, 6–10 seeds each, on a skewed-mode task built so that
diversity *should* pay. **None of them beats a tuned baseline.**

- A **random** bonus beat the novelty bonus at matched λ ([E1b](experiments/E1b-results.md)) —
  the diversity signal contributes nothing.
- **σ×1.5 + lr×0.7 beat every mechanism** (+0.13 mean, +1.56 on the rare mode), because a bonus
  only mimics a smaller step size while also adding variance ([E1c](experiments/E1c-results.md)).
- **There was never any collapse to fix** ([E1e](experiments/E1e-results.md)): effective rank
  drifts 92 → 65 of 128 members and flattens; the DvD volume is flat from generation 50.

The structural reason, which is the durable part: **ES resamples its population every
generation**, so *population* diversity cannot collapse the way a persistent GA's can. What
collapses is the **update direction**. DEGA's machinery exists to stop a small persistent
population converging onto itself — a problem EGGROLL does not have.

Standing decision:
[0002](../claude/decisions/0002-diversity-mechanisms-not-yet-warranted.md).

## What E0 settled

Run on 2026-08-12, code in `code/`, results in [experiments/E0-results.md](experiments/E0-results.md).

- **Concentration confirmed to three significant figures.** Measured pairwise cosine spread
  `0.0109` at `d = 8,144` and `0.0035` at `d = 81,328`; predicted `1/√d` is `0.0111` and `0.0035`.
  It gets *worse* with scale, as theory says. At `d = 10^9`: `3·10⁻⁵`.
- **Every parameter-space metric is at the floor** (0.00–0.10 against a ceiling of 0.87).
- **Per-example fitness vectors work and cost nothing** (0.76–0.87).
- Two lessons that outlive the specific proposals: *project along the objective, don't sketch the
  outputs*; and *spread is not signal*.
