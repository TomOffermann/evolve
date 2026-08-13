# P2 — Structured / partitioned perturbation

**Status:** Open. Tom's idea, 2026-08-12. Worked out as
[Proposal B](../proposals/B-modular-partition-diversity.md).

## Statement

Standard ES perturbs the whole network at once. Instead: split the network into `P` parts,
and let each rollout perturb only a **subset** of parts, holding the rest fixed. The partition
itself can be fixed (per layer, per head, per matrix), or learned / adapted by higher-level
parameters.

## Why this is more than a trick

**1. It restores credit assignment.**
Global ES tells you "this direction in `R^10^9` was good." Useless for understanding. Partitioned
ES tells you "perturbing block 7 was good" — a signal with `P` dimensions instead of `d`, which
you can actually accumulate, plot, and act on. Maintain a per-part utility and run a bandit over
parts: parts that keep paying off get more sampling budget. That *is* the higher-level optimisation
Tom is after, and it's about 40 lines of code.

**2. It multiplies the effective swarm.**
For small `σ` and *disjoint* parts `a, b`:

```
F(W + E_a + E_b) ≈ F(W + E_a) + F(W + E_b) − F(W)
```

The cross term is second-order. So one rollout that perturbs `P` disjoint parts simultaneously
carries information about `P` separate hypotheses — if you can decode them. That's the route
from "30–100 EGGROLL rollouts" to a genuinely swarm-scale signal without a proportional compute
bill. (Decoding is the open part: it's a sparse-recovery problem. See Proposal B.)

**3. It hands the discrete genotype back.**
This is the part that surprised me and it's why P2 and [P1](P1-continuous-diversity.md) belong
in the same conversation. A partitioned individual has a **two-part genotype**: a mask
`m_i ∈ {0,1}^P` (*where* it was perturbed) plus a continuous direction (*how*). And `m_i` is a
bitstring of length `P ≈ 10²–10³`.

**Hamming distance applies to it directly. Unmodified. DEGA's actual metric, not an analogue.**

So the hybrid: Hamming on the mask for structural diversity, a continuous score for diversity
*within* a part. The hard problem shrinks to a much smaller space.

**4. Modularity is a real evolutionary phenomenon and it has a known driver.**
Clune, Mouret & Lipson (*The evolutionary origins of modularity*, Proc. R. Soc. B 2013): modularity
emerges under direct selection pressure to **reduce connection cost**, and the resulting networks
are more *evolvable*, not just tidier. Our analogue of connection cost is **epistasis between
parts** — how much the effect of perturbing `a` depends on what happened to `b`. Measure it:

```
I(a,b) = F(W + E_a + E_b) − [F(W + E_a) + F(W + E_b) − F(W)]
```

Then *learn the partition*: merge parts with high `|I|`, split parts with low internal
interaction. Now "different parts of the brain" is not a metaphor — it's the fixed point of a
measurable procedure.

**5. Interleaving.** Tom's own framing, and it maps onto DEGA's phase structure exactly:

| DEGA phase | Our version |
|---|---|
| Exploitation (`f(x¹) < f(x²)`) | Global full-network EGGROLL. Raw progress. |
| Diversity (`f(x¹) = f(x²)`) | Partitioned swarm, members chosen for max mask-Hamming. Structural coverage. |

## Prior art we're not reinventing

Block-coordinate zeroth-order optimisation is an active line in the LLM-finetuning world and
it works, which de-risks the mechanics considerably:

- **MeZO-BCD** — block-coordinate perturbation for large LLMs.
- **Sparse MeZO** (arXiv:2402.15751) — ZO applied to a chosen parameter subset; *fewer parameters,
  better performance*.
- **SubZero** / *Zeroth-Order Fine-Tuning of LLMs in Random Subspaces* (ICCV 2025), **LOZO**
  (low-rank directions), **AGZO** (activation-derived directions), **HiZOO** (curvature-aware).
- **Dominant-Layer ZO** (arXiv:2606.05516) — claims a *single* layer dominates ZO finetuning.
  If true, this is strong evidence for P2 and also a warning: the learned partition may collapse
  onto one block.
- **ZO Fine-tuner** (arXiv:2510.00419) — *learns* adaptive per-block perturbation variances.
  Closest existing thing to the "higher-level parameters" idea; read before building.

None of these combine block structure with a diversity mechanism or with a learned partition.
That gap is ours.

## Known risks

- **Slower raw convergence.** Restricting to a block reduces the rank/coverage of each update.
  There must be a global-vs-partitioned budget split; guessing it wrong will look like the idea
  failing when it's the schedule failing.
- **Estimator bias.** If part `p` is sampled with probability `π_p`, the ES gradient estimator
  needs a `1/π_p` correction to stay unbiased. Easy to get wrong, easy to not notice, and it
  will silently poison the bandit. Write the correction down before writing the code.
- **Noisy interaction estimates.** `I(a,b)` is a difference of four noisy evaluations. Needs
  common random numbers (same data batch, same rollout seeds) or it's pure noise.
- **Collapse.** The bandit may concentrate on one part (cf. Dominant-Layer ZO). Needs a floor
  on `π_p`, which is itself a diversity mechanism — the snake eats its tail, pleasantly.
