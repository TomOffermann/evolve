# Proposal B — Modular Partition Diversity (MPD)

**Status:** Open. The research bet. Formalises Tom's idea from [P2](../problems/P2-structured-perturbation.md).
**Solves:** [P1](../problems/P1-continuous-diversity.md) partially, [P2](../problems/P2-structured-perturbation.md) fully.
**Injection points:** all three — sampling, selection, and fitness shaping.

## The intuitive idea

Don't shake the whole network at once. Split it into parts and shake **one part at a time**
(or a sparse subset), holding everything else fixed.

Now every individual has a name: *"the one that poked block 7."* And that changes the problem,
because the genotype is no longer a single opaque vector in `R^10^9` — it's a pair:

```
individual = ( WHERE it was perturbed , HOW it was perturbed )
              m_i ∈ {0,1}^P             A_i, B_i restricted to the masked parts
```

`m_i` is a bitstring of length `P ≈ 10²–10³`. **Hamming distance applies to it directly — DEGA's
actual metric, not an analogue.** The impossible continuous-diversity problem shrinks to a
tractable one: Hamming for structural diversity across parts, [Proposal A](A-functional-signature-diversity.md)'s
signature for diversity *within* a part.

And the biological framing is real, not decorative. Brains have regions because regions can be
improved semi-independently. If we force our perturbations to be regional, we're applying the
selection pressure that produces modularity instead of hoping it emerges.

**One-line judgement:** if you can tell me *which part of the network* an individual is exploring,
you have a diversity signal — and you didn't need a metric in 10^9 dimensions to get it.

## Mechanics

### 1. Partition

`P` parts. Start dumb, get smarter:
- v0: one part per weight matrix (or per layer). `P ≈ 10²` for a mid-size transformer.
- v1: per attention head, per MLP block, per row-block of out-features.
- v2: **learned** — see §4.

### 2. Sample masked perturbations

Each member draws a mask `m_i ~ π` (one-hot, or sparse with `s` active parts) and perturbs only
those parts with the usual EGGROLL rank-`r` noise.

**Write the importance correction down before you write the code.** If part `p` is sampled with
probability `π_p`, the ES estimator needs

```
ΔW^(p) = (α / (N σ π_p)) · Σ_{i : m_i[p]=1} F̃_i · A_i^(p) B_i^(p)ᵀ
```

Without the `1/π_p` you get a biased estimator that silently poisons the bandit in §3, and it
will look like the *idea* failing rather than the *estimator* failing.

### 3. Per-part utility → bandit over parts

Maintain a running utility `u_p` = average `|F̃_i|` (or fitness improvement) over members that
perturbed part `p`. Sample `π ∝ softmax(u/τ)` **with a floor** `π_p ≥ π_min`.

This is the "higher-level parameters" from P2: a `P`-dimensional learned object controlling where
evolutionary effort goes. It's also, for free, an **interpretability artefact** — a live map of
which parts of the network are currently learnable. Plot it over training; that plot alone may be
worth the experiment.

*Prior art warning:* Dominant-Layer ZO (arXiv:2606.05516) claims a single layer dominates
zeroth-order LLM finetuning. If that generalises, `π` will collapse onto one part. The floor
`π_min` is not optional. Note that the floor is itself a diversity mechanism.

### 4. Learn the partition via epistasis

The Clune/Mouret/Lipson result: modularity emerges under direct selection pressure to reduce
**connection cost**, and the resulting networks are more *evolvable*. Our analogue of connection
cost is **interaction between parts**:

```
I(a,b) = F(W + E_a + E_b) − [ F(W + E_a) + F(W + E_b) − F(W) ]
```

Zero interaction = the parts are independent = a good partition boundary. Then:
**merge** parts with persistently high `|I|`; **split** parts with low internal interaction.
The partition becomes the fixed point of a measurable procedure rather than a hand-drawn guess.

*Must use common random numbers* — same data batch, same rollout seeds across all four terms —
or `I` is a difference of four noisy numbers and you're clustering noise.

### 5. Where the swarm gets bigger

For small `σ` and disjoint parts, `F(W + E_a + E_b) ≈ F(W+E_a) + F(W+E_b) − F(W)`: the cross term
is second order — which is precisely what `I(a,b)` measures, so we can *verify* the assumption
rather than assume it. When it holds, one rollout that perturbs `s` disjoint parts carries
information about `s` hypotheses at once. Decoding them from the aggregate fitness is a sparse
recovery problem: with random `s`-sparse masks, per-part effects are recoverable by least
squares / compressed sensing over `N` rollouts.

That is the concrete route from "30–100 EGGROLL rollouts" to a swarm-scale structural signal
without a proportional compute bill.

## The DEGA transplant — this is where it gets clean

Tom's interleaving suggestion maps onto DEGA's phases exactly:

| DEGA | MPD |
|---|---|
| Exploitation phase (`f(x¹) < f(x²)`) | **Global** full-network EGGROLL. Raw progress, full-rank updates. |
| Diversity phase (`f(x¹) = f(x²)`) | **Partitioned** swarm. Select members maximising mask-Hamming — structural coverage. |
| Max-Hamming pair selection | Literally unchanged. Hamming on `m_i`. |
| DiPEC's `1/λ` mask subsampling | Subsample the *active parts* of the improving mask with prob `1/λ`. Same operator, same `(1 − 1/(2λ))` distance-retention law. |

DEGA's operators transfer **without modification** in the mask space. That is the strongest
argument for this proposal and it's a consequence of Tom's structural idea, not of the metric work.

## Why it should work

- Restores credit assignment: `P`-dimensional signal instead of `d`-dimensional noise.
- Block-coordinate ZO is *known to work* (MeZO-BCD, Sparse MeZO — "fewer parameters, better
  performance"), so the mechanics are de-risked. Nobody has combined it with a diversity mechanism
  or a learned partition. That gap is the contribution.
- Produces an interpretability artefact for free, which is a stated goal in the project README.
- Modularity has a known evolutionary driver and we can apply it directly rather than wish for it.

## How it could fail

- **Slower raw convergence.** Restricting to a block reduces update coverage per step. There will
  be a global/partitioned budget split, and getting it wrong will masquerade as the idea failing.
  Run the ablation early.
- **Estimator bias** from missing `1/π_p`. See §2.
- **Noisy `I(a,b)`.** Needs common random numbers; even then it's four noisy evaluations.
  If the interaction matrix looks like white noise, §4 is dead and we fall back to a fixed partition
  (which is still a perfectly good proposal).
- **Bandit collapse** onto one dominant part.
- **The partition may not matter.** If transformer blocks are functionally interchangeable enough
  that `I(a,b)` is uniform, there's no modular structure to find. That is a real and interesting
  negative result, and cheap to obtain.

## First experiment

Small model, fixed per-layer partition. Compare three arms at equal compute: (i) global EGGROLL,
(ii) partitioned-only, (iii) interleaved global/partitioned on DEGA's phase rule. Log the per-part
utility `u_p` over time and the interaction matrix `I`. Even if all three arms tie on fitness, the
`u_p` trace and `I` heatmap tell us whether there is modular structure to exploit — which is the
actual question.
