# E3 — JEPA on pendulum: can ES drop the anti-collapse hacks?

**Run:** 2026-08-13 · `code/experiments/e3_jepa_pendulum.py` · 4 arms × 3 seeds × 600 generations ·
2,304-parameter encoder+predictor, 8-dim latent · raw output in `E3-results.txt`.

Tests [P3](../problems/P3-objectives-and-hardware.md)'s claim: stop-gradients and EMA target
encoders exist to stop **gradient descent** finding the collapse shortcut. With ES the fitness is
a black box, so anti-collapse can be stated directly — including non-differentiable terms.
**No arm uses a stop-gradient or an EMA target encoder.**

Scored only on things a collapsed encoder cannot fake. Never on the fitness — a collapsed encoder
achieves a perfect prediction loss.

## Result

| arm | eff_rank (max 8) | sem | probe R² | sem | latent std |
|---|---|---|---|---|---|
| prediction only | 5.24 | 0.32 | 0.724 | 0.002 | **0.078** |
| + variance hinge | 2.48 | 0.00 | 0.725 | 0.001 | 1.204 |
| + variance + covariance | 4.00 | 0.05 | 0.712 | 0.017 | 0.564 |
| **+ non-differentiable rank bonus** | **6.72** | 0.24 | 0.702 | 0.036 | 0.084 |

## Findings

**1. The non-differentiable term wins at the thing it targets.** The rank bonus is the effective
rank of the latent batch — an SVD followed by an entropy. You cannot backprop through it in any
natural way. ES simply adds it to the fitness, and it produces the highest effective rank by a
clear margin (6.72 of a possible 8, vs 2.48–5.24 for the differentiable surrogates). This is the
concrete version of P3's argument: *state what you want rather than a differentiable proxy for
it.*

**2. The variance hinge has an unintended consequence.** It does its job on the metric it
targets — latent std goes to 1.20, exactly the hinge threshold — but effective rank *drops to
2.48*, the worst of any arm. It makes individual dimensions high-variance while leaving them
correlated. VICReg pairs variance with covariance for precisely this reason, and here the pairing
recovers eff_rank to 4.00. Optimising a proxy hard gets you the proxy.

**3. Scale collapse and information collapse dissociate.** The prediction-only arm shrinks the
latent scale by ~15× (std 0.078 vs 1.20) while probe R² stays flat at 0.72. Note that linear-probe
R² is scale-invariant, so it *cannot* see scale collapse; latent std can. They measure different
failures and should both be reported.

## Honest limits — the control did not fully fail

**The prediction-only arm scale-collapses but does not information-collapse in 600 generations.**
Probe R² is 0.724, statistically indistinguishable from every other arm. So the headline claim —
*ES lets you drop the asymmetry hacks* — is **supported directionally but not demonstrated**,
because the control never reached the failure the hacks exist to prevent.

Reaching it likely needs a harder prediction problem, a longer run, or a higher-capacity encoder
where the constant solution is easier to find. Until then the honest statement is finding 1: a
non-differentiable objective term is optimisable by ES and beats differentiable surrogates at its
own target.

## Two design bugs this experiment cost, both instructive

**Normalising prediction error by latent variance is itself an anti-collapse mechanism.** The
first version divided prediction error by the latent variance to "stop shrinking from buying a
better score". That makes the objective scale-free — so the prediction-only control *could not
collapse*, and the experiment proved nothing. (The latents exploded to std ≈ 400 instead.) Now
off by default.

**With the action visible, the pendulum is deterministic and collapse is never the attractor.**
Accurate prediction is easy, so there is no incentive to cheat. Masking the action makes the
torque unobserved and the future irreducibly uncertain: the best honest predictor still carries
error while a constant encoder scores exactly zero, so collapse strictly wins. Real JEPAs collapse
precisely when prediction is hard. `mask_action=True` is now the default.

Both bugs share a shape: **the control was accidentally prevented from failing.** A control that
cannot fail makes every other arm look successful.
