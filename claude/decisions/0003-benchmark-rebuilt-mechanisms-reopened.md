# 0003 — Benchmark rebuilt; the Tier-0 novelty bonus is warranted after all

**Date:** 2026-08-13 · **Status:** Settled. **Supersedes
[0002](0002-diversity-mechanisms-not-yet-warranted.md).**

## Decision

For **sparse verifiable reward** (RL-style post-training — EGGROLL's actual case):

1. **EGGROLL + `ResolutionRule` for σ** — self-adapting step size. Matches a hand-swept σ without
   the sweep.
2. **+ Tier-0 novelty bonus at λ ≈ 0.2–0.4.** It costs ~0.006 pass@1 (within noise) and buys
   **+0.05 pass@16** (+2.9 sem), and it beats a matched random control by +2.7 sem.
3. **Monitor both detectors** — `CollapseDetector` (ES population) and `PolicyDiversityDetector`
   (the policy's own output distribution). Only the second one collapses.

For **dense reward with no measured collapse**, 0002's conclusion still holds: no mechanism beats
a tuned baseline, and step size dominates.

## Why 0002 was wrong

0002 concluded from E1–E1e that no diversity mechanism is warranted, because a random bonus
matched novelty and plain hyperparameters beat both. **That was an artefact of the benchmark.**

E0/E1 ran on a synthetic grammar with dense log-likelihood reward. Diversity collapse is a
failure of *sparse-reward* optimisation — pass@1 up, pass@k down. The grammar task could not
exhibit it, so it could not distinguish "the diversity signal is useless" from "there is nothing
here for any diversity mechanism to do". E1e even measured the second explanation directly and I
reported it, then still generalised the conclusion beyond what the benchmark supported.

The reasoning and controls in E1b/E1c were sound. They were applied to a task that could not
decide the question.

## The evidence that overturned it

Countdown post-training ([E2](../../research/experiments/E2-results.md)): 3 numbers and a target,
emit `a op b op c`, reward 1 if it hits the target. 94k-parameter policy, SFT'd for format only.
92% of problems admit several correct expressions, so pass@k is meaningful.

**The benchmark reproduces the failure:** post-training takes pass@1 from 0.055 → 0.124 while
pass@16 *falls* from 0.211 → 0.164.

**And the signal separates from noise** ([E4](../../research/experiments/E4-results.md), 8 seeds):

| arm | pass@1 | pass@16 |
|---|---|---|
| baseline | 0.1206 | 0.1602 |
| novelty λ=0.2 | 0.1157 | **0.1880** |
| RANDOM λ=0.2 (control) | 0.1152 | 0.1479 |

Matched pass@1; +0.040 pass@16 at +2.7 sem. The random control lands *below* baseline — noise
injection is worse than doing nothing here.

## Other things the rebuild established

- **The classical 1/5th rule inverts under sparse binary reward.** Success rate is non-monotone
  in σ, and at small σ 95% of the population *ties* with the parent — too timid to resolve
  anything, which the rule misreads as too aggressive. Replaced by `ResolutionRule`, which
  controls the **tie rate** (monotone in σ, so well-posed). Needs a dead-band for dense fitness,
  where ties are structurally zero.
- **DiPEC does not port**, now shown properly by sweeping λ rather than condemning it at one
  value: monotone degradation, no sweet spot. In ES the population *is* the gradient estimate, so
  subsampling it just multiplies variance.
- **A non-differentiable objective term works** ([E3](../../research/experiments/E3-results.md)):
  a latent effective-rank bonus (SVD + entropy) beats every differentiable surrogate at its own
  target, 6.72 of 8 vs 2.48–5.24.
- **Two bugs invisible under dense reward:** `rank_normalise` broke ties by argsort order,
  injecting noise in proportion to sparsity; and without **common random numbers** the ES
  gradient is estimated almost entirely from rollout sampling noise, which made training run
  *backwards*.

## The rule this earns — the third one

Alongside "spread is not signal" (E0) and "beat a tuned baseline, not a default one" (E1c):

> **A negative result is only as general as the benchmark that produced it.** Before concluding
> that a mechanism does not work, verify the benchmark can exhibit the failure the mechanism
> targets. If the control cannot fail, nothing you learn from it transfers.

That failure mode recurred three times in this project — E1's task could not collapse, and both
E3 design bugs consisted of accidentally preventing the control from failing.
