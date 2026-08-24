# New Research Directions — Post-E9

**Status:** Implemented, testing on Euler. Four new directions building on the E9 result (partitioned perturbation resists degradation at 0% decline vs 8–11% for global methods).

**Date:** 2026-08-24

---

## Background: What E9 Established

The CPU benchmark (96 runs, 6 arms × 16 seeds) showed:

- **Partitioned perturbation is the only method that doesn't degrade** over 300 generations. IID drops 11% from peak pass@16; partitioned holds flat.
- The mechanism: fewer parameters perturbed per generation → less cumulative drift → better preservation of pretrained capabilities.
- All arms reach similar *peaks* (~0.26 pass@16). The difference is what happens after: partitioned holds, everything else decays.
- Selective update (utility gating at P=3) adds little — the partition is too coarse for the gating to bite.

Four new directions extend these findings.

---

## Direction A — Guided Subspace Sampling

**Status:** Novel combination. SGES (Liu et al., IJCAI 2020) proved the variance reduction theorem for subspace-biased ES. Applying it in EGGROLL's factored format with inter-generation SVD accumulation is new.

### Core idea

Standard EGGROLL throws away all information between generations — every generation samples perturbations from scratch. But the gradient direction the optimizer moved in last generation is informative about where to explore next.

**Store recent gradient directions. Bias future perturbations toward them.**

### Mathematics

In EGGROLL, the ES gradient estimate at generation $t$ is:

$$\Delta W^{(t)} = \sum_{i=1}^{N} w_i \cdot \frac{1}{\sqrt{r}} A_i B_i^\top$$

where $w_i$ are fitness-based weights (rank-normalised), $A_i \in \mathbb{R}^{m \times r}$, $B_i \in \mathbb{R}^{n \times r}$ are the low-rank factors.

**Step 1: Extract the dominant direction.** Compute the rank-1 SVD of $\Delta W^{(t)}$:

$$\Delta W^{(t)} \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top$$

Store the pair $(\mathbf{u}_1, \mathbf{v}_1)$. Maintain a buffer of the last $k$ such pairs.

**Step 2: Bias perturbation factors.** For generation $t+1$, each member $i$ draws random factors $(\tilde{A}_i, \tilde{B}_i)$ as usual, then blends in a stored direction:

$$A_i = (1 - \alpha) \tilde{A}_i + \alpha \cdot \mathbf{u}_j \cdot \sqrt{m}$$
$$B_i = (1 - \alpha) \tilde{B}_i + \alpha \cdot \mathbf{v}_j \cdot \sqrt{n}$$

where $j = i \bmod k$ cycles through stored directions, and the $\sqrt{m}$, $\sqrt{n}$ scaling matches the entry magnitude of random Gaussian factors ($\tilde{A}$ has entries $\sim \mathcal{N}(0,1)$, while $\|\mathbf{u}\| = 1$ so entries $\sim O(1/\sqrt{m})$).

**Step 3: Adaptation.** $\alpha \in [0, 1]$ controls the exploitation–exploration tradeoff:
- $\alpha = 0$: pure isotropic (standard EGGROLL)
- $\alpha = 1$: pure replay of historical directions
- Default: $\alpha = 0.5$

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $k$ | 8 | Number of historical gradient directions stored |
| $\alpha$ | 0.5 | Blend strength (higher = more exploitation) |

### Properties

- **Variance reduction** (SGES theorem): Sampling in the subspace spanned by recent gradients reduces the variance of the gradient estimator, with bias bounded by the projection error.
- **No importance correction needed**: The blend is intentionally biased toward the gradient — this is exploitation, not unbiased estimation.
- **Memory cost**: $k$ pairs of vectors per parameter matrix. For a 7B model with 200 layers: $k \times 200 \times (m + n)$ floats. At $k = 8$, $m = n = 4096$: ~50 MB. Negligible.
- **Compute cost**: One rank-1 SVD per parameter matrix per generation (cheap — $O(mn)$ for the thin SVD, and $\Delta W$ is already computed).

### What's novel

The SGES paper samples in a subspace spanned by recent gradient estimates and proved the variance reduction. What's new here:

1. **Factored format**: SGES operates on full gradient vectors. We extract the dominant direction from $\Delta W = \sum w_i A_i B_i^\top$ (a sum of rank-1 terms) and inject it back into the $(A, B)$ factor space. This keeps EGGROLL's memory-free constraint (no full-rank matrices stored).
2. **Per-parameter-matrix subspaces**: Each weight matrix has its own subspace buffer, allowing different layers to be guided by different historical directions.
3. **Blend instead of project**: SGES samples some perturbations inside the subspace and others orthogonally. We blend every perturbation, which avoids the population-splitting problem with chunked evaluation.

### Risk

$\alpha = 0.5$ may be too aggressive — early local results show pass@16 = 0.152 (below baseline 0.22). The guided directions may collapse diversity by making all perturbations too similar. Lower $\alpha$ (0.1–0.2) or adaptive $\alpha$ could help.

---

## Direction B — Perturbation Archive

**Status:** Novel. MAP-Elites (Mouret & Clune, 2015) maintains an archive indexed by behavioral descriptors. Adapting it to EGGROLL's factored format — storing individual $(A_i, B_i)$ factor pairs indexed by fitness — has not been done.

### Core idea

After each generation, store the factor-pairs of high-fitness members. In future generations, seed a fraction of the population from archived directions instead of from scratch.

**Warm-start from what worked before, with noise for continued exploration.**

### Mathematics

**Archive update** (after pass 1 of generation $t$): Let $\mathcal{T}_k$ be the indices of the top-$k$ members by fitness. For each $i \in \mathcal{T}_k$, store:

$$\mathcal{A} \leftarrow \mathcal{A} \cup \{(A_i, B_i, f_i)\}$$

Trim to archive size $S$ by removing lowest-fitness entries.

**Seeded draw** (generation $t+1$): Every $K$-th member by global index ($K = \lceil 1/\rho \rceil$ where $\rho$ is the seed fraction) draws from the archive:

$$A_i = A_{\text{arch}} + \eta \cdot \tilde{A}_i, \quad B_i = B_{\text{arch}} + \eta \cdot \tilde{B}_i$$

where $(A_{\text{arch}}, B_{\text{arch}})$ is selected from $\mathcal{A}$ by cycling through entries, and $\eta$ is the noise scale controlling exploration around the archived direction.

The remaining $(1 - \rho) \cdot N$ members draw standard isotropic factors.

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $S$ | 64 | Archive size (number of stored perturbation directions) |
| $\rho$ | 0.25 | Fraction of population seeded from archive |
| $\eta$ | 0.3 | Noise scale around archived directions |
| $k$ | 8 | Number of top members archived per generation |

### Properties

- **Temporal persistence**: Discovered good directions survive across generations without maintaining a persistent population. This bridges EGGROLL's ephemeral-population design with GA-style elitism.
- **Memory cost**: $S$ entries × (sum of factor sizes per member). At rank 1, 94k-param model: ~500 floats per entry. $S = 64$: 128 KB. At 7B model: ~50 KB per entry (just the SVD of each layer's perturbation), $S = 64$: 3 MB. Negligible.
- **No importance correction**: Archived members are drawn from a different distribution than isotropic Gaussian, making the ES gradient estimator biased. This is intentional — the archive biases toward high-fitness regions of perturbation space.

### What's novel

MAP-Elites archives *solutions* (full parameter vectors) indexed by *behavioral descriptors*. We archive *perturbation directions* (factored rank-1 terms) indexed by *fitness*. The distinction matters because:

1. We don't need to store full model weights (impossible at LLM scale). Factor-pairs are tiny.
2. The archive represents directions of improvement, not absolute positions in weight space.
3. Seeded members are perturbations *of the current model*, not snapshots of past models.

A natural extension (not yet implemented): index the archive by **behavioral signature** (the double-centred per-example fitness vector from Tier-0) to ensure archived directions are diverse, not just high-fitness.

### Risk

Without behavioral indexing, the archive may converge to a set of similar high-fitness directions, losing the diversity benefit. The noise scale $\eta = 0.3$ may be insufficient to maintain exploration if the archive is homogeneous.

---

## Direction C — ES for MoE Router Optimisation

**Status:** Novel application. EEP (Liu et al., 2024) used evolutionary strategies for expert *pruning*. Using EGGROLL-style low-rank ES for continuous router weight *optimisation* with per-expert credit assignment via partitioned perturbation has not been done.

### Core idea

Routing in Mixture-of-Experts is inherently discrete: the router computes scores $\mathbf{s} = W_r \mathbf{h}$ and selects the top-$k$ experts via $\text{argmax}$. Gradient methods must relax this (Gumbel-Softmax, straight-through estimators), introducing bias or requiring all experts to be activated for gradient computation (eliminating the efficiency advantage of sparsity).

**ES evaluates the hard routing decision directly via fitness. No relaxation needed.**

### Mathematics

**Setup.** A pre-trained MoE model with $K$ expert FFNs $\{E_1, \ldots, E_K\}$, a shared encoder, and a router $W_r \in \mathbb{R}^{K \times H}$. All parameters except $W_r$ are frozen.

**Global ES.** Perturb $W_r$ with low-rank noise:

$$W_r^{(i)} = W_r + \frac{\sigma}{\sqrt{r}} A_i B_i^\top, \quad A_i \in \mathbb{R}^{K \times r}, \; B_i \in \mathbb{R}^{H \times r}$$

Evaluate each perturbed router via hard routing (argmax) and task fitness. Compute the ES update as usual.

**Partitioned ES** (the novel variant). Split $W_r$ into $K$ rows, one per expert. Each member $i$ perturbs only one expert's routing row:

$$W_r^{(i)}[e_i, :] = W_r[e_i, :] + \frac{\sigma}{\sqrt{r}} \mathbf{a}_i \mathbf{b}_i^\top$$

where $e_i$ is the expert assigned to member $i$. This gives **per-expert credit assignment for free**: the fitness of members perturbing expert $e$ tells you how much expert $e$'s routing matters.

**Per-expert utility:**

$$u_e = \frac{1}{|\{i : e_i = e\}|} \sum_{i : e_i = e} |w_i|$$

where $w_i$ are the fitness-based weights. High $u_e$ means expert $e$'s routing is learnable and impactful.

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $N$ | 64 | Population size |
| $\sigma$ | 0.05 | Perturbation scale |
| $\alpha$ | 0.02 | Learning rate |
| $r$ | 1 | Perturbation rank |
| Generations | 150 | Number of ES generations |

### Experimental results (toy model)

4-expert MoE on modular arithmetic (530k params, 300 router params):

| Method | Accuracy | Load balance | Collapsed experts |
|--------|----------|-------------|-------------------|
| Gradient (Adam) | 23.8% | 0.45 | 2 |
| ES global | **50.5%** | 0.63 | 1 |
| ES partitioned | 47.5% | 0.60 | 1 |

The gradient arm fails because soft-routing training optimises a different objective (cross-entropy with softmax weights) than hard-routing evaluation (argmax accuracy). ES evaluates the actual discrete decision.

### What's novel

1. **ES for router weight optimisation** (not just pruning). EEP uses ES to search over binary pruning masks. We optimise continuous router weights via low-rank perturbation.
2. **Partitioned perturbation as per-expert credit assignment.** Each member perturbs one expert's routing row → fitness decomposes into per-expert contributions. This is a signal gradient methods cannot provide without activating all experts.
3. **No auxiliary loss.** Standard MoE training requires hand-tuned load-balancing losses to prevent expert collapse. ES doesn't need them — balanced routing emerges from fitness pressure if balance helps task performance.
4. **Post-training router calibration.** After standard gradient-based MoE training, ES can re-optimise the router for hard-routing accuracy, potentially recovering performance lost to the soft→hard routing gap.

### Connection to literature

- **HARC (June 2026):** showed the routing landscape is sharp — small perturbations to router weights cause catastrophic misrouting under softmax. ES explores this landscape via direct evaluation rather than local gradients.
- **"Routers Learn the Geometry of Their Experts" (May 2026):** router and expert weights are geometrically coupled. Perturbing the router while keeping experts frozen isolates the routing contribution.
- **DeepSeek-V3:** moved to auxiliary-loss-free load balancing via per-expert bias terms. ES-based routing is an alternative path to auxiliary-loss-free routing.

### Risk

The toy model is tiny (4 experts, 300 router params). At real scale (e.g., Mixtral 8×7B with 8 experts per layer × 32 layers), the router has ~131k params per layer. Still small for ES, but the evaluation cost (full model forward pass per member) is high. Feasibility at scale requires efficient batched evaluation.

---

## Direction D — Fine-Grained Partitioning

**Status:** Novel extension. The E8/E9 result established partitioned perturbation at $P = 3$ (emb, hidden, output). Extending to $P = 8$–$16$ via sub-tensor partitioning — splitting weight matrices into row-blocks — has not been tested.

### Core idea

E9 showed partitioned perturbation resists degradation. The selective update (arm 3) was supposed to improve on it by freezing low-utility parts, but it barely helped. The hypothesis: **$P = 3$ is too coarse for utility gating to discriminate.**

With finer partitions, each part represents a smaller, more coherent subregion of the parameter space. The selective mechanism has more parts to choose from, and each part's utility signal is more informative.

### Mathematics

**Sub-tensor partitioning.** Given a parameter matrix $W \in \mathbb{R}^{m \times n}$, split it into $s$ row-blocks:

$$W = \begin{pmatrix} W_{[0:b]} \\ W_{[b:2b]} \\ \vdots \\ W_{[(s-1)b:m]} \end{pmatrix}, \quad b = \lfloor m/s \rfloor$$

Each block is a separate "part" for the partitioned sampler. A member assigned to part $p$ perturbs only the rows of $A_i$ corresponding to that block:

$$A_i[\text{rows of part } p] = \tilde{A}_i[\text{rows of part } p]$$
$$A_i[\text{all other rows}] = 0$$

Since zeroing rows of $A$ zeroes the corresponding rows of $E_i = \frac{1}{\sqrt{r}} A_i B_i^\top$, only the assigned block of $W$ is perturbed.

**Auto-partition.** Blocks are allocated proportional to parameter count. For TinyPolicy (94k params):
- `emb` (7×32 = 224 params): 1 block
- `h` (192×128 = 24,576 params): 6 blocks at $P=8$, 14 blocks at $P=16$
- `out` (7×192 = 1,344 params): 1 block

**Importance correction.** With uniform selection over $P$ parts and $n_\text{active} = 1$:

$$\pi_i = \frac{1}{P}, \quad \text{importance} = \frac{1}{\pi_i \cdot P} = 1$$

For non-uniform priors or $n_\text{active} > 1$, the correction follows the same formula as the standard `PartitionedSampler`.

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $P$ | 8 or 16 | Number of parts (auto-allocated across tensors) |
| $n_\text{active}$ | 1 | Number of parts perturbed per member |

### Properties

- **Sparser updates**: At $P = 16$, $n_\text{active} = 1$, each member perturbs ~6.25% of parameters (vs 33% at $P = 3$). This directly reduces per-generation parameter drift.
- **Finer credit assignment**: The per-part utility diagnostic has 16 dimensions instead of 3, providing a more informative signal about which subregions of the model are useful.
- **Diminishing returns**: At very high $P$, each part contains too few parameters for the perturbation to have a measurable effect on fitness. The signal-to-noise ratio per member drops. The right $P$ balances sparsity against signal.

### Early local results (1 seed)

| arm | P | pass@1 | pass@16 |
|-----|---|--------|---------|
| partitioned (E9) | 3 | 0.079 | 0.248 |
| fine_partitioned | 8 | 0.082 | **0.250** |
| fine_partitioned | 16 | 0.070 | 0.231 |

$P = 8$ matches the best result from 96 Euler runs on a single seed. $P = 16$ may be overshooting — too-fine partitions reduce per-member signal.

### What's novel

Partitioned perturbation at $P = 3$ (one part per parameter tensor) was validated in E8/E9. Sub-tensor partitioning — splitting weight matrices into row-blocks for arbitrary $P$ — is new. The key engineering contribution is the auto-partition algorithm that allocates blocks proportional to parameter count, ensuring each part has a meaningful number of parameters.

At real LLM scale (e.g., 32 transformer layers), $P$ naturally equals the number of layers (or 2× for attention + FFN). Sub-tensor partitioning is needed only when the number of parameter tensors is smaller than the desired $P$.

---

## Comparison of Directions

| Direction | Injection point | Novelty | Risk | Memory overhead |
|-----------|----------------|---------|------|-----------------|
| A: Guided subspace | Sampling | SGES + EGGROLL factored format | Alpha too aggressive → diversity collapse | ~$k \times (m+n)$ per layer |
| B: Archive | Sampling | MAP-Elites for factor-pairs | Archive converges → no diversity | ~$S \times 500$ floats |
| C: MoE routing | Application domain | ES for continuous router weights | Toy scale → may not transfer | None (router is small) |
| D: Fine partition | Sampling | Sub-tensor row-block partitioning | Too-fine $P$ → noise | None |

---

## Key References

- Liu, Li, Qian. *Self-Guided Evolution Strategies with Historical Estimated Gradients.* IJCAI 2020. (Direction A: variance reduction theorem)
- Choromanski et al. *From Complexity to Simplicity: Adaptive ES-Active Subspaces for Blackbox Optimization.* NeurIPS 2019. (Direction A: active subspace learning)
- Mouret & Clune. *Illuminating search spaces by mapping elites.* 2015. (Direction B: MAP-Elites archive)
- Liu et al. *EEP: Efficient Expert Pruning.* 2024. arXiv:2407.00945. (Direction C: evolutionary expert selection)
- EvoESAP. *Non-Uniform Expert Pruning with Evolutionary Search.* March 2026. arXiv:2603.06003. (Direction C: evolutionary MoE structure)
- CoPES. *Cooperative Coevolution for Resource-Constrained Agentic LLM Post-Training.* August 2026. arXiv:2608.02391. (Direction D: validates partitioned perturbation at LLM scale)
- Qiu et al. *Evolution Strategies at Scale: LLM Fine-Tuning Beyond RL.* ICML 2026. arXiv:2509.24372. (EGGROLL baseline)
- arXiv:2601.20861. *Evolutionary Strategies lead to Catastrophic Forgetting in LLMs.* (The forgetting objection that partitioned perturbation addresses)
