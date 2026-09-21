# F2 — EvoCompose: Evolutionary Composition of a Diversified Expert Library, with Memory

**Part of:** F — The Late Evolutionary Layer (`research/base/base.tex` has the shared notation and the general theory)
**Status:** Proposal · no results yet · **Date:** 2026-09-21
**One-line pitch:** *Train a library of small adapters that are good at different things **and deliberately disagree where it matters**, then let an evolution strategy find, per task, how to mix them, only in the last layers so the whole population shares one forward pass. Remember every mix.*

This is the **flagship** of direction F, because it exercises every part of the Evolve thesis:

- slow, diverse components (the library);
- a fast evolved combination (the genome);
- memory (the archive);
- an ablation that tests whether *diversity of the components* is what makes fast evolutionary adaptation possible.

---

## 0. Reading guide

- §1 background: LoRA, the "model soup / task arithmetic" view, and why mixing adapters works at all;
- §2 the idea;
- §3 library construction with the diversity objective, formalising the "agreeable vs non-agreeable questions" note;
- §4 the composition genome, its search and its compute structure;
- §5 memory and the continual stream;
- §6 prior work in detail (LoraHub, Transformer², model merging);
- §7 the benchmark, in two stages;
- §8 hypotheses, risks and kill criteria;
- §9 extensions.

---

## 1. Background

### 1.1 LoRA adapters

A **LoRA** (low-rank adapter) fine-tunes a frozen weight matrix $W \in \mathbb{R}^{m\times d}$ by adding a trainable low-rank term:

$$W' = W + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r\times d},\ B \in \mathbb{R}^{m\times r},\ r \ll \min(m, d).$$

Only $A$ and $B$ are trained (typically $r = 8$–$16$, applied to attention projections $W_q, W_v$ and sometimes the MLPs). One adapter for a 0.5–1B model is a few MB. **A library of hundreds of adapters on one frozen base is cheap to store and to serve.**

### 1.2 Why mixing adapters works at all

Three empirical facts from the model-merging literature make "compose instead of retrain" plausible:

1. **Task arithmetic** (arXiv:2212.04089). Define a *task vector* $\tau_k = \theta_k - \theta_0$, the fine-tuned minus base weights. Adding scaled task vectors, $\theta_0 + \sum_k \lambda_k \tau_k$, often yields a model that does several tasks at once; negating one removes a behaviour.
2. **Linear mode connectivity** from a shared pretrained initialisation: models fine-tuned from the same base usually lie in one low-loss basin, so linear combinations don't fall off a cliff.
3. **Intrinsic dimension** (arXiv:1804.08838, arXiv:2012.13255). Fine-tuning a pretrained LM succeeds even in a random subspace of a few hundred dimensions. The adaptation problem is small. A good *basis* for that subspace is a library of task-specific adapters, and a mixing vector is a coordinate in it.

The gap between "sometimes works" and "reliably works" is (a) the right coefficients, which require search, and (b) a library whose members span useful, *non-redundant* directions, which requires diversity. EvoCompose attacks both.

### 1.3 Evolution strategies at this scale (quick recap)

The genome is a vector of mixing coefficients, $n = 16$–$256$. At this size **CMA-ES** is the right default: it samples $g_i \sim \mathcal{N}(m, \sigma^2 C)$, ranks by fitness, moves $m$ towards the good ones, and adapts $C$ and $\sigma$. Only ranks matter; it needs no gradients; and the $N$ members of a generation are evaluated in parallel. `base.tex` §2 has the full formulation, and F1 §1.3 has a one-page version.

---

## 2. The idea

```
                        ┌──────── Library  ℰ = {Δ_1, …, Δ_K}  (trained once, slow, diverse) ────────┐
 x ──► frozen base, layers 1 … L−L'  ──►  layers L−L'+1 … L with   W' = W + Σ_k α_{γ(ℓ),k} B_k A_k  ──► ŷ
       (shared by ALL members,                     ▲
        computed once per input)                   │ genome g = (α_{γ,k}) ∈ ℝ^{G×K}, G layer groups
                                                   │
                         CMA-ES on few-shot fitness F_c(g)  ◄── recall from archive 𝓜 = {(κ_j, g*_j)}
```

**Three phases:**

1. **Build** (slow, gradient). Train $K$ LoRA experts on $K$ domains, **with a diversity term** that makes them disagree on inputs where the ensemble is uncertain but agree where the answer is clear.
2. **Compose** (fast, evolutionary). For a new task $c$ with a handful of labelled examples, search the mixing genome $g$ with CMA-ES. Compose only in the **last $L'$ layers**, so that the bottom $L - L'$ layers are computed once and shared by the whole population.
3. **Remember.** Store $(\kappa_c, g^\star_c)$. New tasks start by evaluating the archive, then adapt.

**Intuition.** The library is a vocabulary of skills. The genome is a sentence built from that vocabulary. Diversity makes the vocabulary richer and less redundant, so short sentences say more. Evolution is the natural way to write sentences when the grammar (the loss landscape over mixtures) is bumpy and non-differentiable, e.g. exact-match accuracy.

---

## 3. Library construction: the diversity objective

### 3.1 Setup

There are $K$ domains $\mathcal{D}_1..\mathcal{D}_K$, each with labelled data, and an unlabelled (or weakly labelled) **pool** $\mathcal{D}_{\text{pool}}$ covering mixed inputs. Expert $k$ is the base plus LoRA $\Delta_k$, with predictive distribution $p_k(\cdot\mid x)$ over the answer (for LMs: over next tokens, or over the answer string's likelihood).

### 3.2 Plain objective (the control)

$$\mathcal{L}^{\text{indep}}_k = \mathbb{E}_{(x,y)\sim\mathcal{D}_k}\big[-\log p_k(y\mid x)\big].$$

This is what everybody does, LoraHub's library included: each expert is trained in isolation.

### 3.3 Diversity-regularised objective

$$\boxed{\ \mathcal{L}^{\text{div}}_k = \mathbb{E}_{\mathcal{D}_k}\big[-\log p_k(y\mid x)\big] \;-\; \lambda\; \mathbb{E}_{x\sim\mathcal{D}_{\text{pool}}}\Big[\big(1 - a(x)\big)\; \frac{1}{K-1}\sum_{j\ne k} \omega_{kj}(x)\; \mathrm{JS}\big(p_k(\cdot\mid x)\,\big\|\,\mathrm{sg}[p_j(\cdot\mid x)]\big)\Big]\ }$$

Each piece and why it is there:

- **$\mathrm{JS}$ (Jensen–Shannon divergence)** rather than KL, because it is bounded in $[0, \log 2]$ and symmetric. A KL repulsion can be driven to infinity by one expert putting zero mass somewhere, which is a degenerate way to "disagree".
- **$\mathrm{sg}[\cdot]$ (stop-gradient):** expert $k$ moves away from the others; the others are treated as fixed targets. Train round-robin or in parallel with a one-step lag. This keeps the objective per-expert and parallel.
- **$a(x) \in [0,1]$, the agreeableness.** This is the formal version of "distinguish agreeable and non-agreeable questions":
  - *Labelled pool:* $a(x) = \max_j p_j(y^\star\mid x)$. If some expert already gets the answer right with confidence, the question is agreeable and nobody is pushed off the correct answer.
  - *Unlabelled pool:* $a(x) = 1 - \bar H(x)/\log|\mathcal{Y}|$, with $\bar H$ the entropy of the ensemble mean $\bar p = \frac1K\sum_j p_j$. Inputs where the ensemble is uncertain get repulsion; confident ones don't.
- **$\omega_{kj}(x)$, the expertness weighting** ("balance KL based on expertness"). Push hardest where *both* experts are confident but could be confidently different: $\omega_{kj}(x) = \mathrm{conf}_k(x)\,\mathrm{conf}_j(x)$ with $\mathrm{conf} = 1 - H(p)/\log|\mathcal{Y}|$. The alternative $\omega \equiv 1$ is the ablation.

**What it does.** On in-domain data each expert learns its task (first term). On ambiguous out-of-domain inputs, experts are encouraged to hold *different hypotheses*. A later mixture can then select the hypothesis that fits the new task, rather than choosing among $K$ copies of the same guess.

This is the same principle as **DivDis** (arXiv:2202.03418) and **D-BAT** (arXiv:2202.04414), which train multiple heads that agree on training data and disagree on out-of-distribution data so that a few labelled target examples can pick the right head. Their setting is classification heads; here it is a library of LM adapters for later composition.

### 3.4 For language models: what is $p_k(\cdot\mid x)$?

- **Multiple-choice / classification pool items:** the distribution over options from length-normalised log-likelihoods, exactly as in F1 §3.2. This gives a small $|\mathcal{Y}|$, so the entropy normalisation is clean.
- **Free-form generation:** use the next-token distribution at the first $T_0 = 8$ answer positions, under teacher forcing on a reference answer from the base model, and average token-level JS over positions. This is cheaper and good enough to create diversity.

### 3.5 Cost

$\mathcal{L}^{\text{div}}$ requires, per pool batch, forward passes of all $K$ experts (no backward for $j \neq k$). With $K = 8$ small LoRAs on a 0.5B base, this is ~$K\times$ the forward cost on pool batches. Keep the pool batch at ~25% of the training batch.

---

## 4. Composition: genome, fitness, search

### 4.1 Where to compose: the last $L'$ layers

Let the LoRAs be attached in all layers during library training, as usual. **At composition time, use only their last-$L'$-layer parts.** The first $L - L'$ layers run the plain base model. The ablation "compose in all layers" is included in the benchmark.

**Why:**

- **Compute separability.** For a fixed input, the first $L-L'$ layers are identical for all $N$ members, so they are run once. Every member only pays for the last $L'$ layers. For encoder–decoder models (FLAN-T5), the natural cut is **compose in the decoder only**. The encoder, which processes the long few-shot-free input, then runs once per input, and the decoder produces a short answer per member.
- **Evidence that later layers carry task specificity:** C3PO finds its critical routing layers at the end; many PEFT studies find the top layers most task-specific. This is a hypothesis to test, not an established fact for every model.

### 4.2 Genome

Partition the $L'$ composed layers into $G$ groups $\gamma = 1..G$ (e.g. $G = 2$ or $4$ contiguous blocks). The genome is

$$g = \big(\alpha_{\gamma,k}\big)_{\gamma\le G,\ k\le K} \in \mathbb{R}^{G K}, \qquad n = GK,$$

e.g. $K = 16$ and $G = 4$ give $n = 64$. For a composed weight in layer $\ell$ in group $\gamma(\ell)$:

$$W'_\ell = W_\ell + \sum_{k=1}^K \alpha_{\gamma(\ell),k}\; \frac{\alpha_{\text{lora}}}{r} B_{\ell,k} A_{\ell,k}.$$

**Output-space vs factor-space mixing.** LoraHub mixes the *factors* separately, $\hat A = \sum_k \alpha_k A_k$ and $\hat B = \sum_k \alpha_k B_k$, then uses $\hat B\hat A$. That product contains cross terms $\alpha_j\alpha_k B_j A_k$, so it is **not** the linear combination of the adapters' updates. The version above mixes in *output space*, which is exactly linear in $\alpha$ and interpretable as task arithmetic. Include both as an ablation; the cross terms might even help, since they give a larger function class for the same genome.

### 4.3 Computing all members at once

For an input activation $u \in \mathbb{R}^{d}$ at a composed weight:

$$y_i = W u + \sum_{k} \alpha^{(i)}_{\gamma,k}\, B_k\,(A_k u), \qquad i = 1..N.$$

- $A_k u$ ($K$ vectors of size $r$) depends on member $i$ only through $u$. At the *first* composed layer, $u$ is identical for all members, so it is computed once. In later composed layers $u$ differs per member (the members have diverged), so compute per member: $K$ small matmuls of size $r\times d$.
- The extra cost per member per token is $O(K r (m + d))$, compared to $O(md)$ for the base matmul. With $K = 16$, $r = 16$, $m = d = 1024$ this is about +50% of one matmul, and only in the last $L'$ layers. Batched over members, it is one `einsum`.

This is the same economics as EGGROLL's low-rank perturbations: many members, one set of big weights, and per-member low-rank extras.

### 4.4 Fitness

For task $c$ with few-shot set $\mathcal{R}_c$ of $R = 5$ (LoraHub setting) to 64 examples:

$$F_c(g) = \underbrace{\frac1R \sum_{(x,y)\in\mathcal{R}_c} \mathrm{score}\big(\hat y_g(x), y\big)}_{\text{EM / accuracy / } -\text{CE}} \;-\; \beta\,\frac{\|g\|_1}{n}.$$

- **Exact match (EM)** is what BBH reports. It is non-differentiable, and exactly where search is at home.
- **Negative cross-entropy of the target answer** is a smoother proxy. With CMA-ES being rank-based, it is a free choice; use CE for search if EM is too plateau-heavy with $R = 5$.
- **L1** keeps most coefficients near zero, which reduces overfitting and makes the chosen experts interpretable. LoraHub uses exactly $L + \alpha\sum_i |w_i|$.

### 4.5 Search

- **CMA-ES**, $m_0 = 0$ (or archive recall, §5), $\sigma_0 = 0.3$ in coefficient units, $N = 16$–$64$, $T = 40$–$200$ generations.
- **Partitioned perturbation (from the repo's Decision 0006).** Each member perturbs only one group's coefficients. E10 predicts this improves the update for groups with a *weak* gradient and hurts dominant groups. Here that is directly testable per layer group.
- **Population ensembling.** At the end, the top-$q$ genomes form a free ensemble: majority vote for EM, or mean log-probs.

---

## 5. Memory and continual stream

The archive is $\mathcal{M} = \{(\kappa_j, g^\star_j, \hat F_j)\}$.

- **Key** $\kappa_c$: the mean base-model hidden state at layer $L-L'$ over $\mathcal{R}_c$'s inputs. It is already computed in the shared prefix, so it is free.
- **Recall:** evaluate the $M$ nearest archived genomes on $\mathcal{R}_c$ in one batched call, and start CMA-ES at the best. Optionally initialise $C$ from the empirical covariance of the recalled top-$q$.
- **Store** after adaptation.

**Stream for evaluation:** a fixed random order of tasks (BBH's 27 tasks in Stage A), each seen once, then a **revisit block** of 8 tasks with *fresh* few-shot examples. The key numbers:

- $\text{AUC}_c$ over the first $S$ evaluations, revisit vs first visit;
- "archive-recall only" accuracy on the revisit;
- final average accuracy $\bar A_T$.

Forgetting is zero by construction if recall hits the right key; report key-recall accuracy to show it.

---

## 6. Closest prior work, explained

### 6.1 LoraHub (arXiv:2307.13269, COLM 2024) — Stage A's target

- **Library:** ~200 LoRAs, each trained on one FLAN task, on **FLAN-T5-large**, rank 16. Published on the Hugging Face hub under the `lorahub` organisation.
- **Composition:** for each run, randomly sample **20** LoRAs. Weights $w \in \mathbb{R}^{20}$; factor-space merge (§4.2). Optimised with CMA-ES via nevergrad's *Shiwa* selector, **40 steps**. Objective: few-shot loss + $\alpha \sum_i|w_i|$, with **5 examples** per task.
- **Evaluation:** **BIG-Bench Hard (BBH)**, 27 tasks, exact match. Averages from their paper:

| method | BBH avg EM |
|---|---|
| zero-shot | 27.0 |
| in-context learning (5-shot) | 37.3 |
| LoRA tuning on the 5 examples | 37.7 |
| full fine-tuning | 42.1 |
| LoraHub (avg of 3 runs) | 34.7 |
| LoraHub (best of 3 runs) | 41.2 |

- **Takeaway:** composition beats zero-shot by a lot, and the gap between avg and best shows **huge run-to-run variance**, driven largely by *which* 20 modules were sampled. That variance is the opening for EvoCompose. A searched-over (not random) candidate set, a diversity-built library and population ensembling all attack it. LoraHub also uses far fewer inference tokens than ICL (~112 vs ~598 per example), which is an argument for composition over ICL.

### 6.2 Transformer² (arXiv:2501.06252, ICLR 2025)

- **Library of "expert vectors" via SVF (Singular Value Fine-tuning).** Decompose each weight $W = U\Sigma V^\top$ and learn only a vector $z$ that rescales singular values: $W' = U(\Sigma \odot \mathrm{diag}(z))V^\top$. Each expert $z_k$ (math, code, reasoning, …) is trained with RL. It is tiny: one scalar per singular value.
- **Test-time adaptation, three ways:** prompt-based classification of the task, a classifier, or a **few-shot mixture** $z' = \sum_k \alpha_k z_k$ with $\alpha$ found by **CEM** (cross-entropy method, a simple evolution strategy) on a few examples.
- **Relevance:** the CEM variant is exactly "evolve mixing coefficients over a frozen expert library". SVF experts are an alternative to LoRAs for the library; they are even smaller and compose multiplicatively. There is no memory or continual stream, and no diversity-driven library.

### 6.3 Evolutionary model merging (Sakana: arXiv:2403.13187, CycleQD arXiv:2410.14735, M2N2 arXiv:2508.16204)

- **Evo model merge:** CMA-ES over (i) per-layer merging coefficients of full fine-tuned models (parameter space) and (ii) *which layer of which model* to run in sequence (data-flow space). This produced strong Japanese math and VLM models from off-the-shelf parts.
- **CycleQD:** quality-diversity over merges, cycling which skill is the "quality" while the others act as behaviour descriptors, which keeps an archive of diverse merged agents.
- **M2N2:** evolves merge *boundaries* (split points) between pairs of models, and keeps diversity by **resource competition**: models are rewarded for solving examples few others solve (fitness sharing). It also chooses *attractive* mating partners that complement each other.
- **Relevance:** all three are **offline, one-shot** merges producing one final model. EvoCompose is **online and per task**, with a memory. M2N2's resource-competition fitness is a population-level alternative to §3's diversity loss, and it is worth trying as a library-building mechanism (§9).

### 6.4 Libraries and routing without search

- **Arrow** (arXiv:2405.11157): zero-shot routing among LoRAs by comparing each input's hidden state to the top singular vector of each LoRA. This is the "no-search" baseline for composition.
- **Branch-Train-Merge / Branch-Train-MiX** (arXiv:2208.03306, arXiv:2403.07816): train domain experts separately from a shared seed, then ensemble them or merge them into an MoE. This is library-building by domain split, without an explicit diversity term.
- **AdaMerging** (arXiv:2310.02575): learns merge coefficients by **gradient** on unlabelled test-data entropy. It is the gradient-based counterpart of coefficient search, and a good extra control.

---

## 7. Benchmark protocol

### Stage A — no training, published baselines (start immediately)

**Setup:** FLAN-T5-large (~780M) + the LoraHub modules from the HF hub; BBH (27 tasks) with LoraHub's data splits and **5 examples** per task; exact match. It runs on a single 1080 Ti / 3090.

**Arms (matched budget = number of forward passes on the 5 examples):**

| # | arm | question |
|---|---|---|
| A0 | zero-shot | reference |
| A1 | 5-shot ICL | reference |
| A2 | LoraHub (their code: 20 random modules, Shiwa, 40 steps) | reproduce (≈34.7 avg) |
| A3 | same 20 modules, CMA-ES, output-space mixing, all layers | mixing type + optimiser |
| A4 | A3 but **decoder-only composition** (encoder shared by population) | does compute separability cost accuracy? |
| A5 | A4 with **large population** $N \in \{64, 256\}$ at equal wall-clock | can separability buy better search? |
| A6 | **candidate selection by search**: GA over which 20 of ~200 modules (binary mask genome) + CMA-ES on weights, or a sparse L1 CMA over all ~200 | fixes LoraHub's random-subset variance |
| A7 | A4 + population ensemble (top-5 vote) | free ensembling |
| A8 | AdaMerging-style gradient on the same coefficients (CE on the 5 examples) | **key control:** same parameters, gradient optimiser |
| A9 | random search, same coefficients, own seeds | noise control |
| A10 | stream: A4 + archive recall + revisit block | memory |

**Metrics:**

- average EM over 27 tasks (mean ± sem over ≥5 seeds);
- per-task EM;
- **seed variance** (LoraHub's weak point);
- forward passes to 90% of final;
- stream $\text{AUC}_c$ first visit vs revisit;
- wall-clock.

**Compute:** each task is ~$T\times N\times 5$ short generations, i.e. minutes on a 3090. The full table for 27 tasks × 10 arms × 5 seeds ≈ 1–2 GPU-days.

**Stage A decision:** proceed to Stage B if A4 ≈ A3 (separability is cheap) and at least one of A5/A6/A7 improves on A3 by ≥2 sem.

### Stage B — the diversity hypothesis (the novel part)

**Setup:** Qwen2.5-0.5B-Instruct (base frozen). LoRA rank 8 on $W_q, W_k, W_v, W_o$ and the MLPs, in all 24 layers. $K = 8$ domain experts:

| # | domain | data | answer format |
|---|---|---|---|
| 1 | Countdown arithmetic (the repo's E2 generator, larger numbers) | synthetic | expression |
| 2 | GSM8K-style word problems | GSM8K train | number |
| 3 | science QA | ARC-Easy + SciQ train | option letter |
| 4 | commonsense QA | CommonsenseQA train | option letter |
| 5 | logical deduction puzzles | synthetic (ordering constraints) | option / order |
| 6 | date & time arithmetic | synthetic | date string |
| 7 | string manipulation (sort / reverse / count) | synthetic | string |
| 8 | unit conversion & estimation | synthetic | number |

**Pool** $\mathcal{D}_{\text{pool}}$: an even mix of the 8 domains' inputs plus BBH-style *held-out-format* questions generated by the base model, with no labels needed.

**Two libraries of equal training compute:**

- **Lib-I:** $\mathcal{L}^{\text{indep}}$.
- **Lib-D:** $\mathcal{L}^{\text{div}}$ with $\lambda \in \{0.03, 0.1, 0.3\}$.
- **Controls:** **Lib-R** uses *random* repulsion targets, i.e. the JS term is computed against a random permutation of the pool's other experts' outputs from a different input. It matches the "extra regularisation" effect without meaningful diversity (rule 4). **Lib-ω1** sets $\omega \equiv 1$, the no-expertness ablation.

**Evaluation (held-out tasks, never used in library training):**

- BBH tasks that are closest to, but not identical with, the library domains: `multistep_arithmetic_two`, `date_understanding`, `logical_deduction_*`, `object_counting`, `word_sorting`, `navigate`, `tracking_shuffled_objects_*`.
- Composite synthetic tasks, e.g. a word problem whose answer must be reached with Countdown-style operations, or date arithmetic inside a logic puzzle.
- Protocol: 5 and 32 few-shot examples; EvoCompose (A4 analogue) on each library.

**Metrics:**

- held-out EM;
- **pass@k** of the final population (population diversity → coverage);
- the **library diversity diagnostics**: mean pairwise JS on the pool, and the *effective rank* of the matrix of per-example correctness vectors across experts;
- in-domain accuracy of each expert (the diversity term must not destroy expertise).

**Compute:** library training is 8 LoRAs × (1–2 h on a 4090) × 4 variants ≈ 2–3 GPU-days. Composition evaluation is cheap.

---

## 8. Hypotheses, risks, kill criteria

**Hypotheses.**

- **H1 (Stage A):** decoder-only composition (A4) is within 1 EM point of all-layer composition (A3), while enabling ≥4× larger populations at equal wall-clock.
- **H2 (Stage A):** searched candidate sets (A6) and/or population ensembles (A7) reduce seed variance by ≥2× vs LoraHub and raise the mean.
- **H3 (Stage A):** CMA-ES on EM (A4) ≥ gradient on CE (A8) with 5 examples. With only 5 examples, gradient methods overfit fast, and EM is the real metric.
- **H4 (Stage B, the headline):** Lib-D > Lib-I on held-out composition at matched compute, and Lib-D > Lib-R (the effect is diversity, not regularisation).
- **H5 (Stream):** revisits reach first-visit final accuracy in ≤25% of the evaluations.

**Risks.**

- **Diversity may hurt expertise:** check in-domain accuracy; the $a(x)$ gate exists for this.
- **Five examples are very noisy fitness:** EM on 5 items takes only 6 values. Use CE for search, EM for reporting, plus a larger held-out set.
- **Layer-cut sensitivity:** $L'$ might need to be large for T5; sweep $L' \in \{2, 4, 8, \text{all}\}$ decoder layers.
- **The LoraHub modules' availability or format may have changed on the hub:** check first. The fallback is to train ~30 FLAN LoRAs yourself (hours).

**Kill criteria.**

- If **H4 fails** (Lib-D ≤ Lib-I or ≤ Lib-R at matched compute, over the λ sweep), the diversity claim is dead. EvoCompose then reduces to "a better LoraHub", which is still a workshop-level result via H1–H3 and H5.
- If H1 fails badly (>3 EM points), drop late-only composition and use all-layer composition with EGGROLL-style batching.

---

## 9. Extensions

- **Library growth by evolution:** when the archive shows that a task keeps needing large coefficients on many experts, spawn a new expert (distilling that mixture into one LoRA, then training on). This is "consolidation", the slow outer loop.
- **M2N2-style library building:** resource-competition fitness (reward for solving what few others solve) as an alternative to the JS diversity loss.
- **SVF experts** (Transformer²) instead of LoRAs: smaller, multiplicative, and possibly better conditioned for mixing.
- **Per-input genomes:** a key-conditioned genome, as in F1 G2: $\alpha(x) = \alpha_0 + U P \bar e(x)$.
- **Combine with F1:** in an MoE base, route *between* LoRA experts with an evolved router (MoLE-style).
