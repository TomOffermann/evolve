# 0002 — Commit to tuned selection pressure + a collapse detector. No diversity mechanism yet.

**Date:** 2026-08-13 · **Status:** Settled, with an explicit condition for revisiting.

## Decision

For the EGGROLL substrate, the committed approach is:

1. **Vanilla EGGROLL.** No novelty bonus, no niching, no phase controller, no archive.
2. **Sweep σ and the learning rate.** These two knobs dominate every diversity mechanism tested.
3. **Log the Tier-0 collapse detector every generation.** Free.
4. **Add a diversity mechanism only when the detector fires.**

## Why — the evidence

Eight mechanisms were benchmarked across four experiments (E1, E1b, E1c, E1d), 6–10 seeds each.

| finding | experiment |
|---|---|
| Novelty bonus beats baseline (+0.07 mean, +1.01 worst-mode) | [E1](../../research/experiments/E1-results.md) |
| …but a **random** bonus at matched λ beats it | [E1b](../../research/experiments/E1b-results.md) |
| …and **σ×1.5 + lr×0.7 beats everything**, +0.13 mean / +1.56 worst-mode | [E1c](../../research/experiments/E1c-results.md) |
| …because there was **never any diversity collapse** to fix | [E1e](../../research/experiments/E1e-results.md) |

Five of the eight mechanisms were *worse than doing nothing*. The three that helped, helped by
partially mimicking a smaller step size — and doing that directly is strictly better, because a
bonus reduces the effective step **and** adds orthogonal variance, where lowering `lr` only does
the first.

## The structural reason, which is the part worth remembering

**ES resamples its population from scratch every generation.** The noise distribution is fixed
and isotropic, so *population* diversity cannot collapse the way a persistent GA population can.
E1e measured this directly: effective rank drifts 92 → 65 of 128 members and then flattens; the
DvD volume is flat from generation 50 on.

What *can* collapse is the **update direction**. If 70% of the data points the same way, the
aggregate step points that way, and no amount of reweighting a diverse population changes it.

This is the distinction the whole programme was missing. DEGA's machinery exists to stop a small
persistent population from converging onto itself. EGGROLL has no persistent population, so most
of that machinery is answering a question we do not have.

## What this does *not* say

- It does not say diversity is irrelevant to ES in general. It says that on a task where the
  detector shows no collapse, mechanisms cost more than they return.
- It does not retire [P1](../../research/problems/P1-continuous-diversity.md). P1's measurement
  problem was **solved** ([E0](../../research/experiments/E0-results.md): Tier-0 at ρ = 0.76–0.87).
  What E1 retired is routing that measurement into the *update*.
- It does not retire [Proposal B](../../research/proposals/B-modular-partition-diversity.md).
  B is now **better** motivated: it changes what gets perturbed rather than how members are
  weighted, so it acts on update structure — which E1e identifies as the thing that actually
  collapses. Every mechanism that failed was a population-reweighting scheme. B is not one.

## Condition for revisiting

Revisit when the detector fires: `eff_rank` of the Tier-0 signature falling substantially and
staying down, or the DvD volume trending down over a sustained window. Finding a task where that
happens is now the **prerequisite** for any further work on diversity mechanisms — it is a
cheaper and more decisive experiment than building another mechanism.

Candidates worth trying: sparse/binary reward (RLVR-style, where entropy collapse is documented),
genuinely deceptive objectives with a local optimum that must be escaped, and much longer
training runs.

## Two rules this cost us, both now standing

1. **Spread is not signal.** ([E0](../../research/experiments/E0-results.md)) Proposal C produced
   a metric with 18× the discriminative range of the naive baseline and zero predictive power.
2. **Beat a *tuned* baseline, not a default one.** ([E1c](../../research/experiments/E1c-results.md))
   Any intervention that perturbs the update partially mimics a smaller step size, so the
   baseline's step size is a confound in every comparison until it is swept.
