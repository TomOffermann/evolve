# Candidate diversity mechanisms — the long list

Wide sweep for [P1](../problems/P1-continuous-diversity.md). Low bar for entry; the point is
coverage, not quality.

**Six of these have now been measured** — see [E0 results](../experiments/E0-results.md).
Measured `ρ` (rank correlation with held-out behavioural decorrelation; floor 0.00, ceiling 0.87)
is marked **[ρ = …]** below. Unmarked verdicts are still my assessment, not established results.

Cost is per generation, `N` = population, `d` = parameters, `k` = sketch dim, `P` = number of parts.

## A. Genotype space — distances on the parameter delta itself

| # | Idea | Cost | Verdict |
|---|---|---|---|
| 1 | **Cosine / L2 on the full delta `E_i`** | O(N²d) or O(N²k) via sketch | **Dead — measured. [ρ = 0.03–0.10]** Concentration confirmed: cos spread `0.0109` vs predicted `1/√d = 0.0111`, and it worsens with scale. [E0](../experiments/E0-results.md). |
| 2 | **Grassmann / principal angles between `col(A_i)`** | O(N²r²m) | Dead *for random `A`* (same concentration), but becomes meaningful the moment `A` is adapted, biased, or drawn from a learned covariance. Park for later. |
| 3 | **Random-projection (JL) sketch of `E_i`** | O(N·k) | Cheaper than #1 and preserves distances — which is exactly the problem: it faithfully preserves *uninformative* distances. Useful only as plumbing. |
| 4 | **Sign / top-k binarisation → literal Hamming** | O(N·k) | The most direct DEGA transplant: take top-k magnitude coords of `E_i`, binarise, Hamming. **Dead — measured. [ρ = 0.02]** Cute and cheap, but inherits the concentration. Might still work *within a part* (P≈10^4 params); untested there. |
| 5 | **Per-layer energy profile** `(‖E_i^(1)‖, …, ‖E_i^(L)‖)` | O(N·L), ~free | **Dead as a metric — measured. [ρ = 0.00]** Still a genuinely interpretable "where did this member push" descriptor, and the bridge to [P2](../problems/P2-structured-perturbation.md) stands — but it predicts nothing about behaviour on its own. |
| 6 | **Active-subspace projection** `U_tᵀ E_i` | O(N·k·r·m) | **Falsified — measured. [ρ = 0.01–0.02]** at two scales. It *does* fix the spread (18× wider than #1) — and predicts nothing regardless. **Spread is not signal.** → [Proposal C](../proposals/C-active-subspace-diversity.md). |
| 7 | **Part-mask Hamming** | O(N·P), ~free | **→ [Proposal B](../proposals/B-modular-partition-diversity.md).** Hamming, unmodified, on a real bitstring. |

## B. Function / behaviour space — distances on what the member *does*

| # | Idea | Cost | Verdict |
|---|---|---|---|
| 8 | **Per-example fitness vector** `f_i ∈ R^M` over M probe examples | **free** (already computed) | **✅ Validated — measured. [ρ = 0.76–0.87]**, i.e. it attains the ceiling. Best cost/benefit on the list by a wide margin. Double-centre it (remove example difficulty *and* member skill) or you measure "who is good", not "who is different". → [Proposal A](../proposals/A-functional-signature-diversity.md) Tier 0. |
| 9 | **Output-logit delta signature on a fixed probe batch** | 1 shared forward + O(N·k) | **Only if task-projected — measured. [ρ = 0.16 sketched → 0.84 projected]** A JL sketch of output space is near-useless because it faithfully preserves the task-irrelevant variation that dominates. Read along the objective direction instead. → [Proposal A](../proposals/A-functional-signature-diversity.md) Tier 1b. |
| 10 | **Delta-activation signature** `σ(xB_i)A_iᵀ` at a chosen layer | **free** (C4) | **Falsified — measured. [ρ = 0.04]** The free lunch does not work: it captures only the output layer's own contribution, missing everything propagated from earlier layers, and is randomly sketched (see #9). Proposal A Tier 1. |
| 11 | **KL / JSD between perturbed and base output distributions** | 1 shared forward + O(N·V) | Principled for LMs and stochastic policies; `V` (vocab) makes it heavy. Sketch the logits first. |
| 12 | **Behaviour characterisation (NS-ES style)** | rollout-dependent | Uber AI's NS-ES / NSR-ES / NSRA-ES. Works, but requires a *domain-dependent* descriptor — the paper's own stated weakness, and fatal for us since we want to be objective-agnostic. |
| 13 | **Learned/unsupervised descriptors (AURORA)** | VAE training | Autoencode trajectories, use the latent as the descriptor. Removes the domain-dependence of #12 at the cost of a second learning problem inside the loop. Interesting later, not first. |

## C. Population-level aggregation — turning pairwise into one number

| # | Idea | Cost | Verdict |
|---|---|---|---|
| 14 | **Mean pairwise distance** | O(N²) | Naive. Notoriously permits *cycling* — the population can rotate through the same configurations at constant mean distance. |
| 15 | **`log det(K + εI)` — volume of the kernel Gram** | O(N k²) on a k-sketch | **DvD** (Parker-Holder et al., NeurIPS 2020). Geometric reading: the volume of the parallelepiped spanned by the embeddings; maximising it *fills* behaviour space. Fixes cycling. This is the right aggregator. |
| 16 | **k-NN novelty against an archive** | O(N·\|A\|·k) | NS-ES's aggregator. Simple, works, needs archive management. |
| 17 | **Archive-cell occupancy entropy (MAP-Elites)** | O(N) | Needs a discretised descriptor space. Very interpretable. CMA-ME / CMA-MAE are the ES-flavoured versions. |
| 18 | **Adaptive reward–diversity weight (Thompson sampling)** | O(1) | From DvD. The weight schedule is where novelty-bonus methods usually die; DvD adapts it online instead of hand-tuning. Steal this regardless of which metric wins. |

## D. Sampling side — diversity by construction instead of by measurement

Cheapest place to win, and the only place where diversity can *reduce* estimator variance rather
than trading against fitness.

| # | Idea | Cost | Verdict |
|---|---|---|---|
| 19 | **Orthogonal / near-orthogonal MC sampling** | O(N k²) | Choromanski et al., *Structured Evolution with Compact Architectures*. Forcing exploration directions to be exactly orthogonal gives an estimator with **strictly lower MSE** than iid. Not a heuristic — a theorem. Should be in the baseline. |
| 20 | **DPP-structured sampling of perturbations** | O(N k²) | *Structured MC Sampling for Nonisotropic Distributions via DPPs* — generalises #19 and connects it to #15. Same machinery, two injection points. |
| 21 | **Antithetic / mirrored pairs `±E_i`** | free | **Measured worse (−0.18 mean), provisionally.** Classic ES control variate, but at fixed budget it halves the number of *independent* directions (N → N/2) and in high dimension that dominates. Likely why EGGROLL samples independently. [E1](../experiments/E1-results.md). |
| 22 | **Partitioned masks** | free | → Proposal B. |
| 23 | **Learned per-block perturbation variance** | O(P) | *ZO Fine-tuner* (arXiv:2510.00419) learns adaptive per-block variances. Directly the "higher-level parameters" idea from P2. Read before building. |

## E. Mechanism transplants from DEGA (orthogonal to which metric wins)

| # | Idea | Verdict |
|---|---|---|
| 24 | **Phase alternation** (exploit when fitness differs, diversify when tied) | Ports cleanly. "Tied" becomes "fitness spread below threshold". |
| 25 | **Subsampled improving step** (DiPEC's `1/λ` mask) | Continuous analogue: keep a random `1/λ` fraction of the rank-1 components of the aggregated update, or take a partial step. Retains `(1 − 1/(2λ))` of the distance instead of half. Ports cleanly. |
| 26 | **Max-distance tie-breaking** | Needs a metric — this is P1 itself. |
| 27 | **Rejection-until-improvement loop** | DEGA resamples crossover until `f(y) > f(x¹)`. Expensive with LLM rollouts; the ES analogue is to reuse the existing population instead of resampling. |

## Adjacent, unsorted

- **ASEBO** (Choromanski et al., NeurIPS 2019) — learns the active subspace *online* and uses
  contextual bandits to trade sampling inside it vs orthogonal to it. This is Proposal C's
  exploration/exploitation knob, already built. Read closely.
- **Self-Guided ES** (IJCAI 2020) — historical estimated gradients as the subspace.
- **Repulsive deep ensembles** (D'Angelo & Fortuin, NeurIPS 2021) — kernel repulsion in
  *function* space beats repulsion in *weight* space, and they say why. Independent confirmation
  of P1's central argument from a completely different literature.
- **Diversity collapse in RLVR** — entropy collapse, pass@1 ↑ / pass@k ↓, support contraction.
  If we finetune LLMs, this is the failure mode our diversity mechanism should demonstrably
  prevent. Good external benchmark: does ES+diversity keep pass@k where GRPO loses it?
- **Phasic Diversity Optimization** (arXiv:2403.11114) — separates reward and diversity into
  distinct phases. Same shape as DEGA's phase structure, arrived at independently.
