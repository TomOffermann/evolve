# Annotated bibliography

Status: **read** (notes exist) · **skimmed** (abstract + method, enough to place it) · **queued**.

## Substrate

| Status | Work | Why it's here |
|---|---|---|
| read | **EGGROLL** — *Evolution Strategies at the Hyperscale*, arXiv:2511.16652 → [notes](eggroll.md) | Our optimizer. Rank-`r` perturbations, 91% of inference throughput, ~100× vs naive ES. |
| **read** | Qiu et al., *ES at Scale: LLM Fine-Tuning Beyond RL*, arXiv:2509.24372 (ICML 2026) → [dossier](es-vs-rl-evidence.md) | Qwen-2.5 0.5B–14B, Llama-3 1B–8B vs PPO/GRPO/Dr.GRPO. Countdown 66.8% vs 57.5%. **15.5× lower run-to-run std.** **N=30, σ=0.001 fixed across every task and model** while RL needed per-experiment sweeps. |
| **read** | *ES lead to Catastrophic Forgetting in LLMs*, arXiv:2601.20861 → [dossier](es-vs-rl-evidence.md) | **The strongest objection.** ES loses ~10% held-out HellaSwag; GRPO loses nothing. Cause: ES updates are ~1000× higher norm and non-sparse where GRPO's are ~95% sparse. No mitigation proposed — and [Proposal B](../proposals/B-modular-partition-diversity.md) is one. |
| queued | Salimans et al., *ES as a Scalable Alternative to RL*, arXiv:1703.03864 | The origin. Read for the estimator and the parallelisation argument. |

## The template

| Status | Work | Why it's here |
|---|---|---|
| read | *Diversity-Preserving Exploitation of Crossover* (DiPEC / DEGA), arXiv:2507.01524 → [notes](dega-dipec.md) | The idea we're porting. Phase structure + subsampled improving step + max-Hamming tie-break. |

## Diversity — function/behaviour space

| Status | Work | Why it's here |
|---|---|---|
| skimmed | Parker-Holder et al., *Effective Diversity in Population-Based RL* (**DvD**), NeurIPS 2020, arXiv:2002.00632 | `log det` of the embedding kernel = population volume. Fixes cycling. Thompson-sampled diversity weight. The aggregator we're using. |
| skimmed | Conti et al., *Improving Exploration in ES via a Population of Novelty-Seeking Agents* (**NS-ES / NSR-ES / NSRA-ES**), NeurIPS 2018, arXiv:1712.06560 | ES + novelty search. Works, but the behaviour characterisation is domain-dependent — their stated weakness and our blocker. |
| skimmed | D'Angelo & Fortuin, *Repulsive Deep Ensembles are Bayesian*, NeurIPS 2021, arXiv:2106.11642 | Function-space repulsion beats weight-space repulsion. Independent confirmation of [P1](../problems/P1-continuous-diversity.md)'s core argument from Bayesian DL. |
| queued | Cully et al., **AURORA** — *Unsupervised Behaviour Discovery with QD*, arXiv:2106.05648 | Learned descriptors via VAE on trajectories. Removes domain-dependence at the cost of a nested learning problem. |
| queued | *AutoQD*, arXiv:2506.05634 | Automatic behaviour discovery. Same problem, newer. |

## Diversity — sampling side / variance reduction

| Status | Work | Why it's here |
|---|---|---|
| skimmed | Choromanski et al., *Structured Evolution with Compact Architectures*, arXiv:1804.02395 | Orthogonal exploration directions give **strictly lower MSE** than iid. A theorem. EGGROLL samples iid. |
| skimmed | Choromanski et al., **ASEBO** — *Adaptive ES-Active Subspaces*, NeurIPS 2019, arXiv:1903.04268 | Learns the active subspace online; bandit trades inside-subspace vs orthogonal sampling. [Proposal C](../proposals/C-active-subspace-diversity.md)'s backbone. |
| queued | *Structured MC Sampling for Nonisotropic Distributions via DPPs*, arXiv:1905.12667 | Connects orthogonal MC to DPPs — same machinery as DvD, different injection point. |
| queued | *Variance Reduction for ES via Structured Control Variates*, arXiv:1906.08868 | |
| queued | *Self-Guided ES with Historical Estimated Gradients*, IJCAI 2020 | Alternative active-subspace construction. |

## Structured / block-wise zeroth-order

| Status | Work | Why it's here |
|---|---|---|
| queued | **Sparse MeZO**, arXiv:2402.15751 | ZO on a parameter subset: *fewer parameters, better performance*. De-risks [Proposal B](../proposals/B-modular-partition-diversity.md). |
| queued | **MeZO-BCD** / *Elucidating Subspace Perturbation in ZO*, arXiv:2501.19099 | Block-coordinate ZO at scale. |
| queued | *Dominant-Layer ZO*, arXiv:2606.05516 | Claims one layer dominates ZO finetuning. Either strong support for B, or its collapse mode. |
| queued | **ZO Fine-tuner**, arXiv:2510.00419 | *Learns* per-block perturbation variances. Closest existing work to P2's "higher-level parameters". Read before building B. |
| queued | *ZO Fine-Tuning of LLMs in Random Subspaces* (SubZero), ICCV 2025 | |

## Quality-Diversity

| Status | Work | Why it's here |
|---|---|---|
| queued | Mouret & Clune, **MAP-Elites**, arXiv:1504.04909 | The archive idea. |
| queued | Fontaine et al., **CMA-ME**, arXiv:1912.02400; **CMA-MAE** | ES + archive, with emitters. The closest existing "ES with structured diversity". |
| queued | *Phasic Diversity Optimization*, arXiv:2403.11114 | Separates reward and diversity into phases — DEGA's structure, arrived at independently. |

## Modularity / structure

| Status | Work | Why it's here |
|---|---|---|
| skimmed | Clune, Mouret & Lipson, *The Evolutionary Origins of Modularity*, Proc. R. Soc. B 2013, arXiv:1207.2743 | Modularity emerges from **connection-cost pressure**, and yields more *evolvable* networks. The theoretical basis for [P2](../problems/P2-structured-perturbation.md) §4. |
| queued | Mengistu et al., *The Evolutionary Origins of Hierarchy*, PLOS CB 2016 | The follow-up. Hierarchy, same mechanism. |

## Objectives

| Status | Work | Why it's here |
|---|---|---|
| queued | Assran et al., **I-JEPA**; **V-JEPA** | The objective family for [P3](../problems/P3-objectives-and-hardware.md). |
| queued | Bardes et al., **VICReg**, arXiv:2105.04906 | Explicit anti-collapse via variance/covariance. With ES these terms can go straight into fitness. |
| queued | *Connecting JEPA with Contrastive SSL*, NeurIPS 2024 | Why JEPA doesn't collapse. Read before claiming ES removes the need for the asymmetry hacks. |

## Failure modes to benchmark against

| Status | Work | Why it's here |
|---|---|---|
| queued | Diversity/entropy collapse in RLVR; pass@1 ↑ / pass@k ↓ | If we finetune LLMs, this is the failure our diversity mechanism should visibly prevent. Good external benchmark. |
