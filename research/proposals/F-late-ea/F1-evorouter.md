# F1 — EvoRouter: Evolved Late-Layer Routing for Pretrained Mixture-of-Experts

**Part of:** F — The Late Evolutionary Layer (`research/base/base.tex` has the shared notation and the general theory)
**Status:** Proposal · no results yet · **Date:** 2026-09-21
**One-line pitch:** *Freeze a pretrained MoE language model entirely, and let an evolution strategy search a few hundred numbers that nudge which experts the last layers pick. Keep one such genome per domain as a memory.*

---

## 0. Reading guide

This file is meant to be read without opening any paper. It covers:

- §1 the background you need: what an MoE is, why routing is hard to optimise, and what CMA-ES does;
- §2 the idea and its intuition;
- §3 the full math;
- §4 the algorithm and implementation details, including how to make it fast;
- §5 the closest prior work, explained in enough detail to see the gap;
- §6 the benchmark protocol, concrete enough to start on Monday;
- §7 hypotheses, risks and kill criteria;
- §8 extensions.

---

## 1. Background

### 1.1 Mixture-of-Experts (MoE) layers

A transformer block normally contains one feed-forward network (FFN). An **MoE block** replaces it with $K$ FFNs, called **experts** $E_1,\dots,E_K$, plus a small **router**. For every token, the router picks a few experts, and only those are computed. This decouples *total* parameters (large capacity) from *active* parameters (cheap inference).

For a token with hidden state $h \in \mathbb{R}^H$ at MoE layer $\ell$:

$$s_\ell(h) = W_\ell\, h \in \mathbb{R}^K \qquad\text{(router logits, } W_\ell \in \mathbb{R}^{K \times H})$$

$$p_\ell(h) = \mathrm{softmax}\big(s_\ell(h)\big), \qquad \mathcal{S}_\ell(h) = \mathrm{TopK}_k\big(p_\ell(h)\big)\ \text{(indices of the } k \text{ largest)}$$

$$\mathrm{MoE}_\ell(h) = \sum_{e \in \mathcal{S}_\ell(h)} \bar p_{\ell,e}(h)\, E_{\ell,e}(h),$$

where $\bar p$ are the selected probabilities, renormalised over $\mathcal{S}$ or not depending on the model. For OLMoE the flag is `norm_topk_prob`; check the config.

**The model used here: OLMoE-1B-7B** (AllenAI, fully open weights, data and code).

| property | value |
|---|---|
| layers $L$ | 16, all MoE |
| hidden size $H$ | 2048 |
| experts per layer $K$ | 64 |
| active per token $k$ | 8 |
| total / active params | ~6.9B / ~1.3B |
| router params per layer | $64 \times 2048 = 131{,}072$ |

The whole model fits in bf16 on a 24 GB GPU (3090/4090). A 4-bit version runs on a Colab T4; the repo already has this in `code/experiments/moe_routing/run_olmoe.py`.

### 1.2 Why routing is hard to optimise with gradients

$\mathrm{TopK}$ is piecewise constant, so its gradient with respect to the logits is zero almost everywhere. During pretraining this is worked around in two ways:

- gradients flow only through the *selected* experts' weights $\bar p_{\ell,e}$, so non-selected experts get no signal at all;
- auxiliary load-balancing losses keep experts from collapsing.

The consequence is a known **routing gap**. C3PO (§5.1) shows that for OLMoE, an oracle choice of routing weights on the last layers raises average accuracy on six standard benchmarks from ~70% to ~85%, with *no weight changes to any expert*. The experts already contain the knowledge; the router just doesn't send the token to the right place.

An evolution strategy treats the whole model, including the hard top-k, as a black box: it perturbs parameters, runs forward passes and scores the result. It optimises exactly the discrete routing that is used at inference, with no relaxation. In the toy experiment in `code/experiments/moe_routing/` (4 experts, modular arithmetic) this gave 50.5% for ES vs 23.8% for an Adam-trained soft router. The toy is far too small to trust, which is why this idea exists.

### 1.3 CMA-ES in one page

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) searches $g \in \mathbb{R}^n$ by maintaining a Gaussian $\mathcal{N}(m, \sigma^2 C)$.

**One generation:**

1. **Sample** $g_i = m + \sigma\, C^{1/2} \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, I)$, for $i = 1..N$ ($N$ is the population size, default $4 + \lfloor 3 \ln n \rfloor$, often set larger).
2. **Evaluate** the fitness $F(g_i)$; this is the only place the model is touched.
3. **Rank** the members and give the top $\mu = N/2$ positive weights $w_1 \ge \dots \ge w_\mu$ that sum to 1.
4. **Move the mean:** $m \leftarrow m + c_m \sum_{j\le\mu} w_j (g_{j:N} - m)$.
5. **Adapt the covariance** $C$ towards directions that were successful (rank-$\mu$ and rank-one updates), and **adapt the step size** $\sigma$ via the length of an accumulated evolution path.

**Properties that matter here:**

- It is **invariant** to monotone transformations of $F$, because only ranks are used, so accuracy vs log-likelihood does not matter for the dynamics.
- It **learns a metric** $C$: correlated experts get correlated perturbations.
- It works reliably up to $n \approx 10^3$ with a full covariance ($O(n^2)$ memory and update cost). For larger $n$, use sep-CMA (diagonal $C$) or EGGROLL-style low-rank ES.
- The population evaluation is embarrassingly parallel, and here it batches naturally on one GPU (§4.2).

Library: `pycma` (`pip install cma`), or `evosax` (JAX) for fully-on-GPU CMA.

---

## 2. The idea

**Freeze OLMoE completely.** Add a **routing genome** $g$ that shifts the router logits of the last $L'$ MoE layers ($L' = 4$–$5$ of 16). Search $g$ with CMA-ES on a small labelled reference set of the target domain, and score the model *with hard top-k routing, exactly as deployed*. Store the best genome per domain in an **archive**. When a new domain arrives, first evaluate all archived genomes (one batched pass), start from the best, then adapt.

```
 prompt x ──► [ layers 1 … L−L' ]  (frozen, run ONCE per input, cached)
                        │  hidden states h_{L−L'}(x)
                        ▼
      ┌─────────── population of N routing genomes g_1 … g_N ───────────┐
      │  [ layers L−L'+1 … L ]  with logits  W_ℓ h + Δ_ℓ(g_i, x)        │  ← only the suffix is run N times
      └─────────────────────────────────┬───────────────────────────────┘
                                        ▼
                        answer log-likelihoods / accuracy  ──►  fitness F_c(g_i)
                                        │
                           CMA-ES update of (m, σ, C)
                                        │
                           archive  𝓜 = {(κ_c, g*_c, F̂_c)}  (one entry per domain)
```

**Intuition.** The experts are a frozen skill library. The router is a switchboard. We re-wire the switchboard per domain, cheaply and exactly, and remember the wiring. The expert weights never change, so nothing is forgotten, and any domain's wiring can be restored in constant time.

---

## 3. Math

### 3.1 Genome parameterisations

Let $\Lambda = \{L-L'+1,\dots,L\}$ be the adapted layers. Three nested variants, from cheapest to most expressive:

**(G1) Bias-only.** One bias vector per adapted layer:

$$\tilde s_\ell(h) = W_\ell h + b_\ell, \qquad g = (b_\ell)_{\ell\in\Lambda}, \qquad n = K L' .$$

With $K = 64$ and $L' = 4$: $n = 256$, the ideal CMA-ES size. The biases are *input-independent*: "in this domain, prefer experts {17, 42, …} in layer 14". DeepSeek-V3 uses exactly this knob (per-expert bias) for auxiliary-loss-free load balancing, which is evidence that a bias alone can reshape routing meaningfully.

**(G2) Bias + input-conditioned low-rank correction.** Let $\bar e(x) \in \mathbb{R}^H$ be the mean-pooled hidden state of the prompt at layer $L - L'$, which is free to compute. Let $P \in \mathbb{R}^{d'\times H}$ be a **fixed** random Gaussian projection scaled by $1/\sqrt{H}$, with $d' = 8$–$32$:

$$\tilde s_\ell(h, x) = W_\ell h + b_\ell + U_\ell\, P\, \bar e(x), \qquad U_\ell \in \mathbb{R}^{K\times d'}, \qquad n = K L'(1+d').$$

With $d' = 16$ this gives $n = 4352$: use sep-CMA-ES or EGGROLL. The correction lets routing depend on *what the prompt is about* within a domain.

**(G3) Token-position restriction.** Apply the modification only at the **last prompt token** (the one whose next-token distribution is scored), as C3PO does. This is orthogonal to G1/G2 and makes the modified computation even smaller: the suffix is then run for one token per member, not the whole sequence (§4.2).

### 3.2 Fitness

**Reference set.** For context (domain) $c$, a small labelled set $\mathcal{R}_c = \{(x_j, y_j)\}_{j=1}^{R}$ with $R \in \{16, 64, 256\}$. All tasks here are **multiple choice**: $x$ is a question, $y \in \{1..A\}$ is the index of the correct option $a_y$.

**Score per option.** $\text{LL}_g(x, a) = \frac{1}{|a|}\sum_t \log p_g(a_t \mid x, a_{<t})$, the length-normalised log-likelihood as in lm-eval-harness `acc_norm`.

**Two fitness choices:**

$$F^{\text{acc}}_c(g) = \frac{1}{R}\sum_{j} \mathbb{1}\Big[\arg\max_{a} \text{LL}_g(x_j,a) = y_j\Big] \qquad F^{\text{ll}}_c(g) = \frac{1}{R}\sum_j \log \frac{\exp \text{LL}_g(x_j, a_{y_j})}{\sum_a \exp \text{LL}_g(x_j, a)} .$$

$F^{\text{acc}}$ is the true objective but plateau-heavy (it only changes when an answer flips). $F^{\text{ll}}$ is a smooth proxy. Because CMA-ES is rank-based, either is valid. **Default: $F^{\text{ll}}$ for search, $F^{\text{acc}}$ for reporting.** This is still not a gradient method, since $F^{\text{ll}}$ is evaluated under hard routing.

**Leash.** With $n = 256$ and $R = 64$, overfitting is real. Two regularisers:

$$\tilde F_c(g) = F_c(g) \;-\; \beta_1 \frac{\|g\|_1}{n} \;-\; \beta_2\, \frac{1}{R}\sum_{j}\frac{1}{|\Lambda|}\sum_{\ell\in\Lambda} \mathrm{KL}\big(p_\ell(h_j)\,\big\|\,\tilde p_\ell(h_j; g)\big).$$

The KL is between base and modified routing distributions at the scored token. Both terms are computed in the same forward pass, so they are free. Sweep $\beta$; don't fix it.

### 3.3 Why the cost is small (compute separability)

Changing routing in layers $\Lambda$ does not change anything in layers $1..L-L'$. For a scoring task (fixed input, no generation), the prefix activations are **identical across all $N$ members**. With per-layer cost $C$ for one sequence:

$$T_{\text{naive}} = N R\, L\, C, \qquad T_{\text{sep}} = R\,(L-L')\,C + N R\, L'\, C, \qquad \frac{T_{\text{naive}}}{T_{\text{sep}}} = \frac{N L}{(L-L') + N L'} \xrightarrow{N\to\infty} \frac{L}{L'}.$$

For $L=16$, $L'=4$ that is ≈4×. With the **last-token restriction** (G3), the suffix computation per member touches one token instead of the whole sequence of length $T$. Attention in the suffix still needs the keys and values of earlier tokens, but those are *unchanged*, because only the last token's routing is modified. The suffix cost per member therefore drops from $\approx L' T C_{\text{tok}}$ to $\approx L' C_{\text{tok}}$ plus attention over a cached KV. The speedup vs naive becomes $\approx N$ for $N \ll T L / L'$, so the *whole population costs about as much as a few extra forward passes*.

> **Caveat:** answer tokens. For multi-token answers, "last token" means each answer-token position that is scored. Keep G3 exact by modifying routing at *all* answer positions (typically 1–10 tokens) and caching the question prefix. The cost is still $\ll$ naive.

### 3.4 Continual variant: the routing-genome archive

Contexts arrive as a stream $c_1, c_2, \dots$, for example the 6 benchmarks, then MMLU's 57 subjects grouped into ~10 domains. The archive is $\mathcal{M} = \{(\kappa_j, g_j^\star, \hat F_j)\}$.

- **Key.** $\kappa_c = \frac{1}{R}\sum_j \bar e(x_j)$, the mean prompt embedding at layer $L-L'$, which is free.
- **Recall.** Let $\mathcal{M}_M$ be the $M$ archive entries nearest to $\kappa_c$ in cosine distance (or all entries if $|\mathcal{M}|$ is small). Evaluate them all on $\mathcal{R}_c$: this is **one batched population evaluation**, and it is exactly what the population machinery is good at. Initialise $m \leftarrow g^\star_{j^\star}$ with $j^\star = \arg\max_j F_c(g^\star_j)$; set $\sigma$ from the spread of the top-3 recalled genomes, floored at $\sigma_{\min}$.
- **Adapt.** $T$ generations of CMA-ES.
- **Store.** $\mathcal{M} \leftarrow \mathcal{M} \cup \{(\kappa_c, g^\star_c, F_c(g^\star_c))\}$.

**At test time on a mixed stream** (domain unknown), route the *genome* by key: $g(x) = g^\star_{j(x)}$ with $j(x) = \arg\min_j \|\bar e(x) - \kappa_j\|$. The genome then acts as a hard, domain-level "mixture of routers". A soft kernel-weighted blend of genomes is also possible, and that blend is essentially C3PO's kernel-regression surrogate lifted from per-sample to per-domain.

---

## 4. Algorithm and implementation

### 4.1 Pseudocode

```
Input: frozen OLMoE; adapted layers Λ; reference set R_c; archive 𝓜; budget T generations, N members
1   H_pre ← run layers 1..L−L' on all x ∈ R_c (and the answer continuations); cache hidden states + KV
2   if 𝓜 non-empty:                                   # recall
3        G_rec ← nearest-M genomes in 𝓜 by key κ_c
4        F_rec ← EvalPop(G_rec)                          # one batched pass
5        m ← argmax F_rec ;  σ ← max(σ_min, spread(top-3))
6   else  m ← 0 ;  σ ← σ_0
7   for t = 1..T:                                       # adapt
8        g_1..g_N ← CMA.ask()
9        F_1..F_N ← EvalPop(g_1..g_N)                    # batched over members × examples
10       CMA.tell(g, −F̃)                                # minimisation convention
11  g* ← CMA.best ; 𝓜 ← 𝓜 ∪ {(κ_c, g*, F_c(g*))}
12  return g*

EvalPop(G):   # G has shape [N, n]
   for each cached example j:  broadcast H_pre[j] over the N members
   run layers L−L'+1..L with logits W_ℓ h + Δ_ℓ(G[i], x_j)   # member i gets its own Δ
   score answers → per-member, per-example LL → F̃_i
```

### 4.2 Making `EvalPop` fast

- **Member-as-batch.** Stack the members along the batch dimension. The router modification is an additive term $\Delta \in \mathbb{R}^{N\times K}$ that is broadcast per member (G1), or $U_\ell P \bar e$ computed as a batched matmul (G2). The experts' weights are shared, and only which experts each (member, token) pair visits differs. HF's OLMoE implementation already handles arbitrary per-token routing, so a forward hook on the router module that adds $\Delta$ (keyed by batch index) is enough.
- **Prefix cache.** Compute layers $1..L-L'$ once per example. Store the hidden state at layer $L-L'$ and the KV cache. Replicate *views*, not copies, across members.
- **Last-token restriction (G3).** Run the suffix layers only on the scored positions, attending to the cached KV of the prefix tokens. This is a custom forward, but a short one: for the scored positions, call the suffix blocks with `past_key_values`.
- **Memory.** One model copy (~14 GB bf16). Activations: $N \times R \times (\text{scored tokens}) \times H$, tiny.
- **Common random numbers.** Evaluate every member on the *same* reference batch in a generation (all $R$ if $R \le 256$). Different members must never be scored on different examples; the stateful-RNG bug in Decision 0004 was exactly this.
- **Throughput estimate (4090, bf16).** A full forward of a ~200-token prompt is ~1.3B active params × 200 tokens ≈ 0.5 TFLOP, about 5–10 ms. The suffix for 4 layers on ~5 scored tokens is ~1% of that. With $N = 32$, $R = 64$, one generation ≈ 64 prefix passes + 2048 tiny suffix passes, i.e. **well under a second**. 100 generations ≈ minutes per domain.

### 4.3 Hyperparameters (starting points)

| symbol | meaning | start | sweep |
|---|---|---|---|
| $L'$ | adapted layers | 4 | {2, 4, 5, 8} |
| $n$ | genome dim | 256 (G1) | G2 with $d'\in\{8,16,32\}$ |
| $N$ | population | 32 | {16, 32, 64, 128} |
| $\sigma_0$ | initial step | 0.5 × std of router logits | {0.1, 0.5, 1.0}× |
| $T$ | generations | 100 | until plateau |
| $R$ | reference examples | 64 | {16, 64, 256} |
| $\beta_1, \beta_2$ | leashes | 0, 0.1 | log-sweep |

Step size in logit units: measure the typical std of $s_\ell(h)$ over tokens first, and set $\sigma_0$ relative to it. A bias of ~1 std flips top-8 membership frequently; ~0.1 std rarely.

---

## 5. Closest prior work, explained

### 5.1 C3PO — the benchmark target (arXiv:2504.07964)

**What it does.** For each *test* sample, C3PO re-optimises the routing weights $\omega \in \mathbb{R}^{L\times E}$ at the **last token**, in the **last 5 layers** (the "critical layers"), restricted to the **top-20 experts** per layer (the "core experts"). It has no labels for the test sample, so it builds a surrogate from **reference samples**:

- it retrieves the $k = 3$ nearest reference questions, using an external embedding model (NV-Embed-V2);
- it assumes that routing which works on neighbours works on the test sample;
- surrogates include moving $\omega$ towards the neighbours' successful routing (mode-finding / kernel regression), and "neighbourhood gradient descent", i.e. gradient descent on the neighbours' losses;
- it takes 10 optimisation steps with a cosine learning rate.

**Results on OLMoE** (accuracy, baseline → C3PO → oracle):

| | MMLU | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande | avg |
|---|---|---|---|---|---|---|---|
| baseline | 57.8 | 77.9 | 51.3 | 79.8 | 80.7 | 72.2 | 69.9 |
| C3PO | 65.5 | 85.3 | 66.3 | 87.4 | 88.0 | 82.7 | 79.2 |
| oracle | 72.2 | 91.5 | 74.8 | 91.4 | 93.6 | 87.7 | 85.2 |

The **oracle** optimises routing on the test sample using its true label, which is not a legal method; it measures how much headroom routing alone has. Their reference sets come from related datasets: BIG-Bench/SuperGLUE for MMLU, CommonsenseQA/SocialIQA for HellaSwag and PIQA, OpenBookQA/SciQ for ARC, KnowRef for WinoGrande.

**What EvoRouter does differently.**

1. Its fitness is the **real, hard-routed task metric** on labelled references, rather than a neighbour surrogate. It uses no gradient at all, so non-selected experts matter too.
2. It adapts **per domain, then remembers**, instead of per sample and forgetting. The per-sample variant is available via archive keys.
3. Its **population** yields many good routings, so ensembling over the top genomes is free.
4. It has a **continual stream evaluation**, which C3PO does not have.

C3PO's per-sample adaptation is stronger in principle, since it is more specific. The honest question is how much of C3PO's gain a per-domain genome captures, and at what cost. Expect per-domain bias-only (G1) to land *below* C3PO. G2 plus archive-keyed recall is the fair comparison.

### 5.2 Other relevant work (short)

- **DeepSeek-V3 / aux-loss-free balancing (arXiv:2408.15664):** per-expert bias added to the logits *only for selection*, updated by a simple sign rule to balance load. The same knob as G1, used for a different objective.
- **EEP (arXiv:2407.00945):** evolutionary search to prune and merge experts in MoE LLMs. It changes the model; EvoRouter only changes routing.
- **Transformer² (arXiv:2501.06252):** a test-time mix of pre-trained "expert vectors" with coefficients found by CEM (a simple evolutionary method) from a few examples. The same philosophy, applied to a dense model's singular values instead of MoE routing.

---

## 6. Benchmark protocol

### 6.1 Tasks and data

**Primary (comparable to C3PO).** MMLU, HellaSwag, ARC-Challenge, ARC-Easy, PIQA, WinoGrande. Use **lm-evaluation-harness** task definitions and the `acc_norm` metric where it exists (`acc` for MMLU and WinoGrande), so numbers match published OLMoE baselines.

Two reference-set regimes:

1. **In-distribution references.** Take $R$ examples from each benchmark's *train* split (MMLU: `auxiliary_train` or dev; WinoGrande: train; the others have train splits) and evaluate on the official test/validation split. This is the clean setting.
2. **C3PO-style references.** Use the related datasets listed in §5.1. This reproduces their conditions for a direct comparison.

**Continual stream.** Split MMLU's 57 subjects into 10 domain groups (e.g. STEM-math, STEM-physics/chem, bio/medicine, CS, law, economics/business, history, philosophy/ethics, psychology/sociology, misc). Stream order: the 6 benchmarks, then the 10 MMLU groups. Then **revisit**: present 4 earlier domains again, with fresh references, to test recall. Rule 3 applies: without recurring or related contexts the archive cannot show a benefit.

### 6.2 Arms

All adaptive arms get the **same budget of forward passes** on the references, e.g. $T \times N \times R = 100 \times 32 \times 64$.

| # | arm | what it tests |
|---|---|---|
| A0 | base OLMoE | reference point; should reproduce published numbers |
| A1 | 5-shot ICL | the standard no-training adaptation |
| A2 | **gradient on the same genome**: soft routing (softmax-weighted over all experts, or a straight-through top-k), Adam on $b_\ell$ with the same $\mathcal{R}_c$ and leash | **the key control:** same parameters, different optimiser |
| A3 | random search on the same genome (same $N\times T$ samples from $\mathcal{N}(0,\sigma_0^2 I)$, keep best), with its own seed per run | noise control (rule 4) |
| A4 | CMA-ES, G1 bias-only | the main arm |
| A5 | sep-CMA / EGGROLL, G2 conditioned | expressiveness |
| A6 | A4 + G3 last-token only | speed; does restricting positions cost accuracy? |
| A7 | C3PO (reported numbers; their code if it runs) | the published target |
| A8 | archive-recall only (no adaptation) | how much memory alone gives |
| A9 | archive recall + CMA-ES | the full system |
| A10 | LoRA SFT on $\mathcal{R}_c$ (rank 8, attention only) | "just fine-tune" upper-ish reference; it changes weights, so it can forget |

### 6.3 Metrics

- **Accuracy** per task (mean ± sem over ≥5 seeds; seeds vary both reference sampling and search).
- **Sample efficiency:** forward passes to reach 90% of the final gain.
- **Off-domain damage:** apply genome $g^\star_c$ to all *other* tasks, and report the mean change. This shows why per-domain genomes plus a recall key are needed, as opposed to one global genome.
- **Routing statistics:** load entropy $H(\bar p_\ell)/\log K$ per layer, fraction of top-8 sets changed vs base, and the KL leash value.
- **Stream metrics:** $\bar A_T$ (average final accuracy over all domains), BWT (0 by construction if recall is exact; report recall accuracy of the key router), and $\text{AUC}_c$ (fitness integrated over the first $S$ evaluations) for revisited vs fresh domains.
- **Wall-clock and GPU memory.**

### 6.4 Compute budget

- **Per domain:** 100 generations × 32 members × 64 refs, mostly tiny suffix passes. Roughly **5–20 minutes on a 4090**, and 3–5× that on a T4 with 4-bit weights.
- **Full primary study:** 6 tasks × 10 arms × 5 seeds ≈ 300 runs ≈ 1–3 GPU-days. It shards trivially across the 1080 Tis if the 4-bit path works there; check bitsandbytes support for Pascal.

### 6.5 Two-week plan

| day | task |
|---|---|
| 1–2 | Reproduce A0 with lm-eval-harness; measure router-logit std per layer; implement the router hook with a per-batch-index $\Delta$ |
| 3–4 | Prefix cache + `EvalPop`; unit test: $\Delta = 0$ reproduces A0 exactly; bias on one expert changes its load as expected |
| 5–6 | A4 on ARC-C with $R=64$; A2 and A3 on the same setup |
| 7 | **Decision point:** A4 vs A2 vs A3 on ARC-C (5 seeds) |
| 8–10 | All six tasks, A4/A2/A3/A8 |
| 11–14 | Stream + archive (A9); G2; G3 speed arm |

---

## 7. Hypotheses, risks, kill criteria

**Hypotheses.**

- **H1:** Bias-only CMA-ES (A4) improves accuracy over A0 on ≥4/6 tasks, with gains of several points.
- **H2:** A4 ≥ A2 (gradient on the same genome) at matched budget. This is the "hard routing needs a black-box optimiser" claim at real scale.
- **H3:** A4 > A3 by ≥2 sem. The search is doing work, not noise injection.
- **H4:** On revisited domains, A9 reaches A4's final accuracy in ≤20% of the evaluations.
- **H5:** Per-domain genomes damage other domains (off-domain Δ < 0), justifying the key router.

**Risks.**

- **Overfitting a small $\mathcal{R}$:** gains on the references that don't transfer. Mitigate with the leash sweep, $R = 256$, and held-out reporting.
- **Per-domain granularity may be too coarse:** most of C3PO's gain may be per-sample. Mitigate with G2 conditioning and per-sample key recall.
- **Noisy fitness:** with $R = 16$ the rank of members may be mostly noise (cf. E12). Measure split-half reliability of member rankings before trusting anything.
- **Implementation:** the last-token suffix forward with a KV cache is the fiddly part; do G1 without G3 first.

**Kill criteria.**

- If **H2 fails** (A2 ≥ A4 on ≥4/6 tasks at matched budget), routing is not a place where black-box search beats gradients at real scale. Report the negative result and fold routing into F2 as one knob among many.
- If **H3 fails**, stop: the effect is not search.

---

## 8. Extensions

- **Ensembling the final population:** a majority vote over the top-$q$ genomes is a free ensemble. Compare its accuracy (and pass@k for generative tasks) with the single best.
- **Discrete genome:** $g \in \{-1, 0, +1\}^{K L'}$ as *force-exclude / neutral / force-include* per expert per layer, searched by a GA. This connects to F3's theory angle.
- **Generative tasks:** GSM8K / Countdown (the repo's E2 task) with pass@k. Compute separability is lost for generated tokens, but memory separability remains. This is a good bridge to the repo's post-training results.
- **Other MoEs:** DeepSeekMoE-16B (28 layers, 64 routed + 2 shared experts; needs 2×24 GB or 4-bit) and Qwen1.5-MoE-A2.7B.
