"""Rechenberg's 1/5th success rule -- self-adapting step size.

E1c's finding was that step size dominates every diversity mechanism we tried. That
makes hand-tuning sigma both the highest-return knob and an unsatisfying answer: any
mechanism benchmarked against a *fixed* sigma is partly being scored on how badly
that sigma was chosen.

The 1/5th rule removes the confound. Classical ES result: the optimal ratio of
successful mutations is about 1/5. Above it, the step is too timid -- enlarge it.
Below it, too aggressive -- shrink it. Here "success" is the fraction of population
members that beat the current parent's fitness, which is directly available.

With sigma self-adapted, a diversity mechanism has to earn its place against a
baseline that is already tuning itself, which is the fair test.
"""


class OneFifthRule:
    def __init__(self, sigma, target=0.2, factor=1.15, lo=1e-4, hi=1.0, window=10):
        self.sigma = sigma
        self.target, self.factor = target, factor
        self.lo, self.hi = lo, hi
        self.window = window
        self.hist = []

    def update(self, success_rate):
        """success_rate: fraction of the population that beat the parent."""
        self.hist.append(success_rate)
        if len(self.hist) < self.window:
            return self.sigma
        avg = sum(self.hist[-self.window:]) / self.window
        if avg > self.target:
            self.sigma = min(self.hi, self.sigma * self.factor)
        elif avg < self.target:
            self.sigma = max(self.lo, self.sigma / self.factor)
        return self.sigma

    def observe(self, F, parent_fitness):
        """Convenience: compute the success rate from a population's fitness."""
        return self.update(float((F > parent_fitness).float().mean()))


class ResolutionRule:
    """1/5th-rule in spirit, but adapted for **sparse binary reward**, where the
    classical version inverts.

    Measured on countdown (see E2), success rate vs sigma is *non-monotone*:

        sigma    0.001   0.005   0.02    0.1     0.25
        success  0.051   0.375   0.594   0.672   0.047
        ties     0.949   0.594   0.285   0.164   0.082

    Two things break Rechenberg's rule here. First, success is not monotone, so a
    target success rate does not identify a unique sigma -- with a 1/5 target the
    rule walks sigma *up* through 0.1 until it falls off the cliff at 0.25, then
    oscillates. That is exactly what we observed: sigma ranging over 0.08-0.25 with
    training stalled.

    Second and more fundamental: at small sigma, **95% of the population ties with
    the parent**. A tie means the step was too small to change any verifier outcome
    -- too timid to measure. The classical rule reads a low success rate as "too
    aggressive" and shrinks sigma further, which is backwards.

    So control **resolution** instead of success: choose sigma so that a target
    fraction of the population is distinguishable from the parent. That statistic
    *is* monotone in sigma, which makes it a well-posed control target. A tie-rate
    target of ~0.5 lands on sigma ~0.005-0.01, matching the sigma/alpha sweep.

    Under **dense** fitness there are no ties at all, so the tie rate carries no
    information and the rule must HOLD sigma rather than act on a signal that is
    structurally zero. The `dead_band` below enforces that. Getting this wrong is
    not hypothetical: the first version shrank sigma on every step of the dense
    JEPA objective (E3), walking it into a regime where the update diverged.
    """

    def __init__(self, sigma, target_tie=0.5, factor=1.10, lo=1e-4, hi=0.5,
                 window=10, dead_band=0.02):
        self.sigma = sigma
        self.target_tie, self.factor = target_tie, factor
        self.lo, self.hi = lo, hi
        self.window = window
        self.dead_band = dead_band
        self.hist = []

    def observe(self, F, parent_fitness, tol=1e-9):
        tie = float((F - parent_fitness).abs().le(tol).float().mean())
        self.hist.append(tie)
        if len(self.hist) < self.window:
            return self.sigma
        avg = sum(self.hist[-self.window:]) / self.window
        if avg < self.dead_band:
            return self.sigma              # dense fitness: no tie signal, hold
        if avg > self.target_tie:          # cannot resolve differences -> step out
            self.sigma = min(self.hi, self.sigma * self.factor)
        else:                              # resolving plenty -> tighten, stay local
            self.sigma = max(self.lo, self.sigma / self.factor)
        return self.sigma
