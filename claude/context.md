# Current Context

**Last updated:** 2026-08-13

## The benchmark was rebuilt (2026-08-13) — read this before trusting E0/E1

E0/E1 ran on a synthetic grammar with **dense log-likelihood** reward. EGGROLL's real case is
RL-style post-training with **sparse verifiable** reward. The old task could not exhibit
diversity collapse, so it could not test any mechanism meant to prevent it. E1's "no mechanism
helps" was drawn on a benchmark where nothing could have helped.

Two realistic benchmarks now exist and both are in `code/`:

- **E2 countdown post-training** — sparse binary verifiable reward, 94k-param policy, most
  problems have several correct answers. **Reproduces RLVR collapse**: pass@1 0.055 → 0.124,
  pass@16 0.211 → 0.164.
- **E3 JEPA on pendulum** — representation collapse; a **non-differentiable** effective-rank
  bonus beats all differentiable surrogates at its own target.

Consequences: the diversity signal separated from the random control for the first time (E2);
the classical 1/5th rule **inverts** under sparse reward and is replaced by a *resolution rule*
that matches a hand-swept σ without the sweep; DiPEC degrades monotonically with no sweet spot
(E4). Two shipped bug fixes: tied ranks in `rank_normalise`, and common random numbers without
which the ES gradient is estimated from sampling noise and training runs backwards.

## Superseded standing decision — kept for the reasoning, not the verdict

**No diversity mechanism is in use.** E1–E1e (8 mechanisms, 6–10 seeds) found that none beats a
*tuned* baseline, because **there is no diversity collapse to fix**: ES resamples its population
every generation, so population diversity cannot collapse the way a persistent GA's can. What
collapses is the *update direction*, which is a step-size problem.

**Current standing: `decisions/0005-novelty-not-established.md`.** 0003 claimed the novelty
bonus beat its random control by +2.7 sem; that control had a frozen RNG seed and drew identical
noise every run. Against a varying control (E7, 8 seeds) the margin is **+1.1/+1.7 sem** — under
the bar. **No mechanism is established.**

Committed approach: EGGROLL + `ResolutionRule` for σ, both detectors logged. λ=0.4 novelty is the
best measured pass@k setting (0.2007) and may be used deliberately, knowing a random bonus gets
most of the way there. On dense objectives with no measured collapse, step size dominates.

## Active Focus

The project has a concrete research program now. **Settled:** EGGROLL (low-rank ES) as the
optimizer substrate, DEGA/DiPEC as the template for the diversity mechanism
(see `decisions/0001-substrate-and-scope.md`).

**The core open problem:** DEGA's diversity signal is Hamming distance on bitstrings. Our
genotypes are Gaussian parameter deltas in ~10^9 dimensions, where all pairwise distances
concentrate and carry no ranking information. We need an honest continuous replacement.
Full statement: `research/problems/P1-continuous-diversity.md`.

## State of the research program

Structure under top-level `research/` (see its README for the map). **E0 has been run**
(2026-08-12, `research/experiments/E0-results.md`, code in `code/`) and it reordered the
proposals:

- **A — Functional Signature Diversity** — measure what the model *does*. **Tier 0 (per-example
  fitness vectors, free) validated at ρ = 0.76–0.87** against a floor of 0.00. Tier 1 (the
  "free" activation-delta signature) **falsified at ρ = 0.04**.
- **B — Modular Partition Diversity** — Tom's structural idea; perturb parts of the network so
  the genotype regains a discrete mask and Hamming applies *unmodified*. **Untested** — E0's
  negative results don't touch it. Now the only substantive open bet.
- **C — Active-Subspace Diversity** — **falsified as a metric** (ρ = 0.010–0.019 at two scales).
  Survives only as a variance-reduction idea (orthogonal sampling, antithetic pairs), which was
  never tested.

Build order is now **A(Tier 0) → B → C(sampling only)**.

Two lessons that outlive the specific proposals: **project along the objective, don't sketch the
outputs** (same delta: JL sketch ρ=0.16, task-projected ρ=0.84); and **spread is not signal**
(C had 18× the discriminative range of the naive baseline and zero predictive power).

Concentration is now settled empirically, not just asserted: measured pairwise cosine spread
matched the predicted `1/√d` to three significant figures at both 8k and 81k parameters, and
degrades with scale as theory says.

## Open Tasks

- **n≈20-seed rerun of novelty vs control.** At the λ=0.4 effect size that reaches ~2.5 sem;
  ~90 min, the cheapest decisive experiment available. Until then the effect is neither believed
  nor discarded.
- **E8** (running): is Proposal B's +0.094 pass@16 structural, or just the smaller effective step
  that a 1-of-3 partition implies? Global σ sweep spanning the partitioned arm's effective step.
- **Proposal B** (partitioned perturbation) is the best-motivated survivor — it acts on update
  structure rather than population weighting, and update direction is what E1e showed actually
  collapses. Every mechanism that failed was a population-reweighting scheme; B is not one.
- Clean rerun of the antithetic-sampling arm — its "worse than iid" result came from the E1 run
  that carried the RNG-ordering flaw.
- Check the `eggroll-es` PyPI licence and API — reuse vs own implementation.
- Read: ZO Fine-tuner (arXiv:2510.00419) and Sparse MeZO before building Proposal B.
- README.md still has the `[...]` placeholder in "Key Design Areas".

## Four rules the experiments cost us — apply to everything new

1. **Spread is not signal** (E0). A metric can have 18× the discriminative range of the baseline
   and zero predictive power. Test against a held-out behavioural target.
2. **Beat a *tuned* baseline, not a default one** (E1c). Any intervention that perturbs the
   update partially mimics a smaller step size, so the baseline's step size is a confound in
   every comparison until it is swept.
3. **A negative result is only as general as the benchmark that produced it** (0003). Verify the
   benchmark can exhibit the failure the mechanism targets. If the control cannot fail, nothing
   you learn transfers.
4. **A positive result is only as good as its control's ability to beat it** (0005). Check the
   control is free to vary on every axis the treatment varies on. A control with a frozen seed is
   not a control — it is one sample.

Three conclusions have been overturned so far and the cause was the same every time: a control
that could not do its job. A benchmark that could not exhibit the failure (E1); a control that
could not collapse (E3, twice); a control that could not vary (E4). Specific enough to check for
directly, and checking is cheap compared with building on it.

## Notes

- Tom's voice: bold, personal, slightly irreverent, first-person "I" — preserve in all edits.
- "Statistical Illusion (SI)" is a coined term Tom wants to use — treat it as a proper term.
- Research notes live in top-level `research/`. `claude/` holds meta-context and decisions only.
- Tom works at ETH; hardware realities (MacBook, 3090/4090, lots of 1080 Tis) are a genuine
  design constraint, not an afterthought — see `research/problems/P3-...`.
