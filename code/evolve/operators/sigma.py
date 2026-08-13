"""Injection point 3: step-size adaptation.

E1c's headline was that **step size dominates every diversity mechanism tested**,
which makes this the highest-value module in the library and also a trap: any
mechanism benchmarked against a badly chosen fixed sigma is partly being scored on
how badly that sigma was chosen. Self-adapting sigma removes the confound.

All rules take a *paired* parent fitness -- the unperturbed model evaluated under
the same common random numbers as the population. Unpaired, the success/tie
statistic is dominated by sampling noise and the rule random-walks (E2).
"""

from __future__ import annotations

import torch


class FixedSigma:
    """No adaptation. Correct when sigma has been swept; a confound when it hasn't."""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def update(self, fitness, parent_fitness):
        return self.sigma


class OneFifthRule:
    """Rechenberg's 1/5th success rule.

    **Do not use on sparse binary reward** -- measured to invert there (E2). Kept
    because it is the classical baseline and because its failure mode is the reason
    `ResolutionRule` exists.

    Two things break it under sparse reward. Success rate is *non-monotone* in sigma
    (0.05, 0.38, 0.59, 0.67, 0.05 across sigma = 0.001 .. 0.25), so a 1/5 target does
    not identify a unique sigma -- the rule walks sigma up until it falls off the
    cliff, then oscillates. And at small sigma 95% of members *tie* with the parent,
    which means too timid to resolve anything; the rule reads that low success rate
    as too aggressive and shrinks further."""

    def __init__(self, sigma, target=0.2, factor=1.15, lo=1e-4, hi=1.0, window=10):
        self.sigma = sigma
        self.target, self.factor = target, factor
        self.lo, self.hi, self.window = lo, hi, window
        self.hist: list[float] = []

    def update(self, fitness, parent_fitness):
        self.hist.append(float((fitness > parent_fitness).float().mean()))
        if len(self.hist) < self.window:
            return self.sigma
        avg = sum(self.hist[-self.window:]) / self.window
        if avg > self.target:
            self.sigma = min(self.hi, self.sigma * self.factor)
        elif avg < self.target:
            self.sigma = max(self.lo, self.sigma / self.factor)
        return self.sigma


class ResolutionRule:
    """Control the **tie rate** instead of the success rate. Recommended default.

    Same spirit as the 1/5th rule -- adapt the step from how the population responds
    -- but on a statistic that is monotone in sigma and therefore well-posed. A tie
    means the step was too small to change any outcome, i.e. too timid to *measure*;
    a target tie rate of ~0.5 lands on the same sigma a hand sweep finds.

    E2: it settles at sigma ~ 0.005-0.016 on countdown post-training and beats the
    hand-swept fixed sigma on pass@1 (0.124 vs 0.106), with no sweep.

    `dead_band` handles dense fitness, where ties are structurally zero and the
    statistic carries no information: the rule must HOLD rather than act on a signal
    that is always zero. The first version lacked this and shrank sigma every step on
    the dense JEPA objective, walking it into a divergent regime (E3)."""

    def __init__(self, sigma, target_tie=0.5, factor=1.10, lo=1e-4, hi=0.5,
                 window=10, dead_band=0.02, tol=1e-9):
        self.sigma = sigma
        self.target_tie, self.factor = target_tie, factor
        self.lo, self.hi, self.window = lo, hi, window
        self.dead_band, self.tol = dead_band, tol
        self.hist: list[float] = []

    def update(self, fitness, parent_fitness):
        tie = float((fitness - parent_fitness).abs().le(self.tol).float().mean())
        self.hist.append(tie)
        if len(self.hist) < self.window:
            return self.sigma
        avg = sum(self.hist[-self.window:]) / self.window
        if avg < self.dead_band:
            return self.sigma                     # dense fitness: no signal, hold
        if avg > self.target_tie:
            self.sigma = min(self.hi, self.sigma * self.factor)
        else:
            self.sigma = max(self.lo, self.sigma / self.factor)
        return self.sigma


class CosineDecaySigma:
    """Scheduled decay. Useful as a control against adaptive rules -- if a schedule
    matches an adaptive rule, the adaptation is not earning its place."""

    def __init__(self, sigma, final_frac=0.3, total=1000):
        self.sigma0, self.sigma = sigma, sigma
        self.final_frac, self.total = final_frac, total
        self.t = 0

    def update(self, fitness, parent_fitness):
        import math
        self.t += 1
        p = min(self.t / max(self.total, 1), 1.0)
        f = self.final_frac + (1 - self.final_frac) * 0.5 * (1 + math.cos(math.pi * p))
        self.sigma = self.sigma0 * f
        return self.sigma
