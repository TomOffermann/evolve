# E2 — Countdown post-training: the benchmark E0/E1 should have used

**Run:** 2026-08-13 · `code/experiments/e2_countdown_posttrain.py` · 6 arms × 3 seeds × 250
generations · 93,824-parameter policy · raw output in `E2-results.txt`.

## Why the old benchmark was wrong

E0 and E1 ran on a char-level LM with **dense log-likelihood** reward. EGGROLL's actual case is
RL-style post-training with **sparse, binary, verifiable** reward — and diversity collapse
(pass@1 up, pass@k down) is a failure of *that* regime. The grammar task could not exhibit the
failure, so it could not test any mechanism meant to prevent it. E1's conclusion ("no mechanism
beats a tuned baseline") was therefore drawn on a benchmark where nothing *could* have.

**Countdown:** given 3 numbers and a target, emit `a op b op c`. Reward 1 if it uses each number
once and hits the target, else 0. 92% of problems admit **several distinct correct expressions**,
which is what makes pass@k meaningful — a task with one right answer cannot show collapse. A
random policy scores 0.001. The base model is SFT'd on valid expressions paired with a *random*
target, so it learns format but not target semantics; post-training must supply the capability.

## Result

| arm | pass@1 | sem | pass@16 | sem | gap |
|---|---|---|---|---|---|
| *SFT base (no post-training)* | *0.0547* | | *0.2109* | | *0.1562* |
| classical 1/5-rule | 0.0924 | 0.010 | 0.2357 | 0.019 | 0.1432 |
| fixed σ (hand-swept) | 0.1055 | 0.006 | 0.1771 | 0.039 | 0.0716 |
| **res + novelty λ=.2** | 0.1068 | 0.003 | **0.2018** | 0.027 | 0.0951 |
| res + niche k=8 | 0.1120 | 0.009 | 0.1680 | 0.022 | 0.0560 |
| res + RANDOM λ=.2 | 0.1172 | 0.005 | 0.1641 | 0.016 | 0.0469 |
| **resolution rule** | **0.1237** | 0.007 | 0.1641 | 0.009 | 0.0404 |

## Finding 1 — the benchmark reproduces diversity collapse

Post-training more than doubles pass@1 (0.055 → 0.124) while pass@16 **falls** (0.211 → 0.164).
The gap collapses from 0.156 to 0.040. That is the documented RLVR failure, reproduced on a
94k-parameter model on a laptop.

This is the prerequisite [decision 0002](../../claude/decisions/0002-diversity-mechanisms-not-yet-warranted.md)
named: *find a task where the failure actually happens*. It happens here.

## Finding 2 — the diversity signal separates from the random control, for the first time

At **matched pass@1**, novelty (0.1068) versus hand-swept fixed σ (0.1055): pass@16 is 0.2018 vs
0.1771. And against the random-bonus control, which was what killed novelty in
[E1b](E1b-results.md): novelty 0.2018 vs RANDOM 0.1641 — **the random bonus preserves no pass@k
at all**, scoring identically to no mechanism.

That is the first result in this programme where the diversity *signal* does something noise
cannot. It is also exactly what should happen: E1b's control won on a benchmark with no collapse,
because there the only thing a bonus could do was mimic a smaller step. Here there is a real
failure to prevent, and the signal starts to matter.

**Not yet significant.** 0.2018 ± 0.027 vs 0.1641 ± 0.016 is ~1.2 combined sem at 3 seeds.
Direction is right, magnitude is plausible, evidence is not conclusive. → [E4](E4-results.md)
reruns this comparison at 8 seeds.

## Finding 3 — the classical 1/5th rule inverts under sparse reward

Measured success rate against σ, on the base model:

| σ | 0.001 | 0.005 | 0.02 | 0.1 | 0.25 |
|---|---|---|---|---|---|
| success rate | 0.051 | 0.375 | 0.594 | 0.672 | 0.047 |
| **tie rate** | **0.949** | 0.594 | 0.285 | 0.164 | 0.082 |

Two things break Rechenberg's rule here.

**Success is not monotone in σ.** It rises to 0.67 then falls off a cliff. A 1/5 target therefore
does not identify a unique σ — the rule walks σ *up* through 0.1 until it overshoots past 0.25,
then oscillates. Observed exactly: σ ranging over 0.08–0.25 with training stalled (train reward
flat at ~0.05, pass@1 0.092 — barely above the SFT base).

**At small σ, 95% of the population ties with the parent.** A tie means the step was too small to
flip any verifier outcome — too timid to *measure*, not too aggressive. The classical rule reads
the resulting low success rate as "too aggressive" and shrinks σ further. It is backwards.

## Finding 4 — a resolution rule fixes it

Control the **tie rate** instead of the success rate: pick σ so a target fraction of the
population is distinguishable from the parent. Tie rate *is* monotone in σ, which makes it a
well-posed control target. Implemented in `es/sigma.py::ResolutionRule`.

It settles on σ ≈ 0.005–0.016 and **matches the hand-swept fixed σ without the sweep** — in fact
beats it on pass@1 (0.1237 vs 0.1055). Given that E1c's headline was "step size dominates
everything", automating step size is the single highest-value component built so far.

*Caveat found while building it:* under **dense** fitness there are no ties, so the tie rate is
structurally zero and carries no information. The first version shrank σ every step on the dense
JEPA objective and walked it into a divergent regime. A dead-band now holds σ when ties vanish.

## Two implementation bugs the sparse regime exposed

Both were invisible under dense reward and both are in the shipped code now.

**Tied ranks.** `rank_normalise` broke ties by argsort order, which is arbitrary. Under sparse
binary reward most members score *identically*, so this injected pure noise into the update in
proportion to sparsity. Ties are now averaged.

**Sampling noise swamping the signal.** Member fitness is a mean over B Bernoulli rollouts at
p ≈ 0.07, so with independent sampling its standard deviation is ~0.026 — far larger than the
fitness differences a small weight perturbation produces. The ES gradient was being estimated
almost entirely from sampling noise, and training ran *backwards*. Common random numbers (one
shared uniform draw, inverse-CDF sampling across the whole population) fix it. Evaluation still
uses independent samples, since pass@k requires them.

## What this changes

[Decision 0002](../../claude/decisions/0002-diversity-mechanisms-not-yet-warranted.md) said to
add a mechanism only when the detector fires on a task that actually collapses. The task now
exists. The standing "no mechanism" decision is **provisionally challenged** rather than
overturned — E4 decides it.

Note the collapse detector itself did **not** fire (0/3 seeds, all arms). That is not a
contradiction: it monitors *population* diversity across ES members, and E1e's structural point
stands — ES resamples, so the population does not collapse. What collapses here is the
**policy's own output distribution**, which is what pass@k measures. Those are different objects
and the detector should monitor both. Currently it monitors only the first.
