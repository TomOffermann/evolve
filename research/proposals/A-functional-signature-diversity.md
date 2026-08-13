# Proposal A — Functional Signature Diversity (FSD)

**Status:** **Tier 0 validated** by [E0](../experiments/E0-results.md) (ρ = 0.76–0.87 vs a
floor of 0.00). **Tier 1 falsified** by the same experiment (ρ = 0.04). Primary metric.
**Solves:** [P1](../problems/P1-continuous-diversity.md).
**Injection points:** selection/archive (primary), fitness shaping (secondary).

> **E0 verdict (2026-08-12).** The cheapest tier wins outright and the clever tiers lose.
> Two corrections to what is written below:
>
> 1. **Tier 1 does not work.** The "free" activation-delta signature scores ρ = 0.04, barely
>    above floor. Kept below for the record with the failure explained inline.
> 2. **The signature must be *projected along the objective*, not randomly sketched.** Same
>    delta, JL sketch → ρ = 0.16; read along the task direction → ρ = 0.84. The output space
>    is mostly task-irrelevant variation and a JL sketch preserves it faithfully — which is
>    the defect. This was not in the original write-up and is the main thing E0 taught us.
>
> Net effect: **use Tier 0.** It is free, objective-agnostic, and attains the ceiling.

## The intuitive idea

Stop asking *"are these two noise vectors different?"* — they always are, and it tells you nothing.
Ask instead: **"do these two mutants disagree with each other on the same inputs?"**

Fix a small probe batch. Run every population member on it. Record what each one actually
*outputs*. That output vector is the member's fingerprint. Two members are similar if they push
the model's behaviour in the same direction, no matter how different their weight deltas look.

This is the honest continuous analogue of Hamming distance, and the analogy is exact:
Hamming counts *in how many positions do these two genotypes disagree*; FSD measures
*on how many probe inputs, and in which output directions, do these two mutants disagree*.
DEGA counts disagreements in genotype; we measure disagreement in behaviour.

**One-line judgement:** if two perturbations make the model answer the same questions the same
way, they are the same individual — however orthogonal their weight vectors happen to be.

## Mechanics, in three tiers of increasing cost

### Tier 0 — per-example fitness vector (cost: zero) — **✅ validated, ρ = 0.76–0.87**

You already evaluate every member. Keep the *per-example* scores rather than only the mean:

```
f_i = (F_i(x_1), …, F_i(x_M)) ∈ R^M        # M = probe examples in the batch
φ_i = double_centre(f)_i / ‖·‖             # then normalise
```

**Double-centring is not optional** (E0): subtract the per-example mean *and* the per-member
mean. The first removes example difficulty, the second removes overall skill. What's left is
what is *distinctive* about this member. Skip it and you measure "who is good", not "who is
different" — which is the one thing a diversity metric must not do.

`φ_i` is a behavioural descriptor. "Which problems did you get right" *is* behaviour. This costs
nothing, works for any objective with per-example structure (LM, JEPA, most RL), and should be
in the codebase before anything more sophisticated.

Limit: coarse. With binary rewards and `M = 64` you get a 64-bit signature — which, pleasingly,
means you can run **literal Hamming distance** on it and DEGA transplants almost verbatim.

### Tier 1 — delta-activation signature (cost: ~zero, exploits C4) — **❌ falsified, ρ = 0.04**

*Kept for the record. E0 found two things wrong with it: it captures only the output layer's
own low-rank contribution (missing everything propagated from earlier layers), and — the fatal
one — a random sketch of output space is nearly uninformative. See Tier 1b.*

EGGROLL's forward pass already computes `Δy_i = σ (x B_i) A_iᵀ` for each member. That is the
member's contribution to the activations at that layer — a rank-`r` functional fingerprint,
free. Take it at the last layer (or a chosen probe layer), flatten over the probe batch,
sketch with a **fixed** random matrix `S ∈ R^{(M·n) × k}`, `k ≈ 64–256`:

```
φ_i = S ᵀ vec(Δy_i) / ‖·‖
```

Because `Δy_i` is rank-`r`, `Sᵀ vec(Δy_i)` is computable from `(xB_i)` and `A_i` without ever
materialising `Δy_i`. Satisfies C1.

**Normalise.** Otherwise `φ_i` is dominated by perturbation magnitude and you'll measure `σ`
instead of behaviour. We want the *direction* of functional change.

### Tier 1b — task-projected delta (cost: ~zero) — **⚠️ works, but decays: ρ 0.84 → 0.20**

Take the **total** output delta `logits_i − logits_base` (the base forward is EGGROLL's shared
matmul, so this is still nearly free), and read it **along the objective direction** rather
than through a random sketch:

```
d_i[b] = Δlogits_i[b, y_b]      # the coordinate the loss actually reads
φ_i    = double_centre(d)_i
```

ρ = 0.838 at initialisation versus 0.157 for the JL sketch of the *same* tensor. That gap is
the whole lesson.

It then decays to 0.20 over training, because it is a *linear* proxy for the objective and the
network saturates. Tier 0 — the actual objective value — holds at 0.76+. So Tier 1b is a good
diagnostic and a bad production metric.

### Tier 2 — full output signature (cost: one shared forward pass) — untested

On the actual outputs (logits / JEPA latents) rather than the delta. Only worth trying **with
an objective-aligned projection**, per Tier 1b; a generic sketch of this will fail the same way
Tier 1 did. Relevant for objectives like JEPA where "the task direction" is less obvious than
a correct-token coordinate — that is exactly where this needs thought, not where it comes free.

## Aggregating to a population score

Pairwise means for a single tie-break; for a population score use the **DvD volume**:

```
K_ij = k(φ_i, φ_j)  (squared-exponential or plain cosine)
D(population) = log det(K + εI)
```

Geometric reading: the volume of the parallelepiped spanned by the fingerprints. Maximising it
*fills* behaviour space. This is why the determinant beats mean pairwise distance — mean distance
permits **cycling** (population rotates through the same configurations at constant mean
distance), the determinant does not.

At `N > 10^3`, don't form `K`. Use greedy DPP-MAP selection on the `k`-dim `φ` (`O(N k²)`), or
subsample a few hundred members per generation. Satisfies C2.

## The DEGA transplant

**Diversity phase** (fitness spread below threshold): among the tied members, keep the subset
maximising `log det K` — greedy DPP selection. This is exactly DEGA's "keep the pair with the
largest Hamming distance", generalised from 2 members to a subset.

**Exploitation phase**: DEGA's insight is that the improving mask carries critical flips *plus*
irrelevant ones, so take only a `1/λ` subsample. The continuous analogue: when a direction
improves fitness, apply only a random `1/λ` subsample of its rank-1 components (or an equivalently
scaled partial step). Expected retained distance follows the same `(1 − 1/(2λ))` law, with cosine
distance in place of Hamming.

**Adaptive weight**: steal DvD's Thompson sampling for the reward/diversity trade-off. Hand-tuned
novelty weights are where NS-ES-style methods go to die.

## Why it should work

- Measures the quantity we actually care about. Everything else is a proxy for this.
- Invariant to the symmetries that make parameter distance meaningless: neuron permutation,
  layer-wise rescaling, null directions.
- **Independent confirmation:** repulsive deep ensembles (D'Angelo & Fortuin, NeurIPS 2021) found
  that kernel repulsion in *function* space works where repulsion in *weight* space fails —
  arrived at from Bayesian deep learning, no connection to ES. Two literatures, same conclusion.
- Objective-agnostic. Unlike NS-ES's hand-designed behaviour characterisations, nothing here is
  RL-specific. Works for JEPA and LM objectives unchanged, which matters for
  [P3](../problems/P3-objectives-and-hardware.md).

## How it could fail

- **Probe batch sensitivity.** Too small or unrepresentative → the fingerprint measures the probe,
  not the model. Mitigation: resample the probe batch on a slow schedule, check that the diversity
  ranking is stable across two independent probe batches. *This is the main risk.*
- **Signature dominated by a few high-variance output dimensions.** Mitigation: whiten `φ` using
  running per-dimension statistics.
- **Tier 0 too coarse** on easy tasks where everyone scores identically. Detect: fingerprint
  variance collapses. Escalate to Tier 1.
- **σ confound** if you forget to normalise. You will forget once.

## Cost summary

| Tier | Extra forwards | Memory | Compute |
|---|---|---|---|
| 0 | 0 | `O(N·M)` | free |
| 1 | 0 | `O(N·k)` | `O(N·k·r)` |
| 2 | 1 shared probe batch | `O(N·k)` | `O(N·k)` + probe forward |

All satisfy C1–C5.

## First experiment — **done**

[E0 results](../experiments/E0-results.md). Tier 0 validated at ρ = 0.76–0.87 against a floor
of 0.00 and a ceiling it attains by construction; Tier 1 falsified; the task-projection
principle discovered. Naive parameter cosine came in at 0.03–0.10 as predicted.

## Next

- ~~Does Tier 0 still work when per-example fitness is **binary**?~~ **Answered by
  [E2](../experiments/E2-results.md)/[E4](../experiments/E4-results.md).** Yes, and it is the
  regime where the signature finally earns its keep: on sparse binary reward, a novelty bonus
  built on Tier 0 preserves pass@k where a matched random bonus preserves none. That is the
  reversal of [E1b](../experiments/E1b-results.md), and it happened because the benchmark
  changed, not the method.
- Binary reward also exposed two implementation bugs invisible under dense reward: tied ranks in
  `rank_normalise` (arbitrary tie-breaking injects noise in proportion to sparsity), and the need
  for **common random numbers** (without them the ES gradient is estimated from sampling noise
  and training runs backwards).
- Objectives without an obvious task direction (JEPA). Tier 1b's projection is easy for a
  correct-token coordinate and genuinely open for a latent prediction loss.
- Wire Tier 0 into the DEGA phase controller and measure whether it changes anything, which is
  the question E0 does *not* answer: a metric that predicts behaviour is necessary, not
  sufficient, for a diversity mechanism that improves optimisation.
