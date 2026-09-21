# F — The Late Evolutionary Layer: Frozen Swarm, Evolving Skin

**Status:** Proposal draft, literature-grounded, nothing run yet · **Date:** 2026-09-21
**Builds on:** Direction C (ES for MoE routing, `code/experiments/moe_routing/`), Decision 0006 (partitioned perturbation), E10/E11 (less drift ⇒ less forgetting), E12 (sampling noise at small σ).

---

## TL;DR

The whiteboard idea, stated in one sentence: **put all the heavy learning into frozen (or slowly trained) components that are run once and shared, and put the evolutionary algorithm on a tiny, late, possibly discrete set of parameters that decides how those components are combined.** EAs are bad at 10⁹ dimensions and great at 10²–10⁴; the architecture should hand them the second problem, not the first.

The literature supports this more than I expected. The pattern has worked many times under different names, but nobody has framed it as a **continual adaptation system**:

- **World Models** (Ha & Schmidhuber, 2018) used a frozen VAE + RNN with an **867-parameter controller trained by CMA-ES**, which is almost exactly this design.
- **LoraHub** (2023) searches ~20 LoRA mixing coefficients gradient-free, and **Transformer²** (2025) uses CEM over expert-vector mixing coefficients at test time.
- **Sakana's evolutionary model merging**, **CycleQD** and **M2N2** evolve merge recipes with only a few parameters.
- **C3PO** (2025) re-mixes the expert weights of a pretrained MoE at test time without gradients and gains 7–15%.
- **FOA** (ICML 2024) runs CMA-ES on a handful of input prompt tokens of a frozen ViT for test-time adaptation, using forward passes only.
- **PathNet** (2017) used a GA to pick paths through frozen modules for continual learning.

What is **missing**, and where Evolve can contribute:

1. treat the tiny genome as **the unit of memory** (an archive of genomes is a hierarchical, zero-forgetting memory);
2. use EA parallelism to run **thousands of candidate genomes per forward pass of the backbone**, exploiting the separability that a late layer buys;
3. use **discrete genomes** where runtime theory for EAs, which is your home turf, actually applies.

I draft five ideas below (four core ones and one stretch). **My recommendation:** start with **Idea 1 (EvoRouter)**, because the code and Colab already exist and C3PO gives a published target to beat. Build **Idea 2 (EvoCompose)** as the main thesis direction. **Idea 3 (Sandwich)** is the one where your theory background turns into a paper nobody else can write.

---

## Part 1 — The high-level idea

### 1.1 The picture

```
            ┌──────────── frozen / slow (gradient, pretrained) ───────────┐
 x ──► φ (backbone / embedders) ──► {f_1, …, f_K} (experts, embedders)  │
            └──────────────────────────────┬──────────────────────────────┘
                                           │  z(x), {f_k(z)}   ← computed ONCE, cached / shared
                                           ▼
                         ┌──── A_g : fast / evolved (tiny genome g) ────┐
                         │  routing, mixing, masking, gating, firing    │  ← N members evaluated in parallel
                         └───────────────────┬──────────────────────────┘
                                             ▼
                               ψ (frozen decoder / head)  ──► y, action, text
```

- **Slow system (φ, f_k, ψ).** Holds all the representational capacity. It is trained by SGD, pretrained, or refreshed rarely. Only **one copy lives in GPU memory.**
- **Fast system (g).** Holds the "what do I do *here*" decision. It is small (10¹–10⁴ numbers, or bits), searched by an EA, and **stored per context** so that it forms the memory.

This is the complementary-learning-systems split (slow cortex, fast hippocampus) with an EA in the hippocampus role. It also answers the whiteboard question *"EA deep or shallow?"*: **shallow and late**, for four reasons below.

### 1.2 Why shallow + late is the right place for an EA

**(a) Dimension.** ES gradient estimates align with the true gradient at roughly $\sqrt{N/d}$. You measured this yourself in E10: 0.037 at N=128, d=94k. At $d = 10^3$ and $N = 10^3$ the alignment is ≈ 1. CMA-ES with a full covariance works up to about $d \sim 10^3$–$10^4$. Below that dimension, **EAs stop being "slow SGD" and become the best tool available**, especially for non-differentiable, discrete and multi-modal objectives.

**(b) Intrinsic dimension says the adaptation problem is small.** Li et al. (2018) and Aghajanyan et al. (2020) show that fine-tuning succeeds in random subspaces of only a few hundred to a few thousand dimensions. Black-Box Tuning (BBT, 2022) runs CMA-ES in a ~500-dim random projection of a prompt. The *adaptation* problem is low-dimensional even when the model is not.

**(c) Separability.** This is the big one, and it has two different flavours that are easy to confuse:

- **Memory separability (always holds).** Population members are genomes, not model copies. One set of backbone weights is shared and members differ only in the tiny late layer. EGGROLL already exploits a weaker form of this.
- **Compute separability (holds only sometimes).** If the backbone's input does not depend on the genome, run the backbone **once per input** and evaluate all N members on cached features. See the cost model in §2.2. This holds for classification, multiple-choice log-likelihood scoring, single-step routing on a fixed prompt, and test-time adaptation. It **fails** for autoregressive generation, where generated tokens depend on the genome, and for closed-loop control, where observations depend on actions. There, members still share memory but each pays its own forward pass.

  → **Design rule:** prefer benchmarks and objectives where compute separability holds, or restrict the EA layer's influence to the part of the computation after the last genome-dependent input.

**(d) Forgetting.** Your own E10/E11 result: partitioned ES reaches the same pass@1 with **~0.5× the parameter drift**. The arXiv:2601.20861 diagnosis blames ES forgetting on dense, high-norm updates. A late evolved layer is the extreme version of this. The backbone **never drifts**, so forgetting on the backbone is zero by construction, and all task-specific change lives in a genome you can store, swap and roll back.

### 1.3 Why this is a step towards "evolve", not just another PEFT method

| capability (README) | how the late-EA design provides it |
|---|---|
| Continual learning without catastrophic forgetting | Backbone frozen; each context gets its own genome; nothing is overwritten |
| Hierarchical memory | Level 0 = weights (slow). Level 1 = genome archive (fast, indexed by context). Level 2 = archive statistics / meta-parameters (which genomes, which mutation distributions work where) |
| Rapid adaptation | Warm-start from the archive: **the archive *is* the initial population**, so evaluating it is one parallel batch |
| Self-modification | Genomes can encode structure (which experts, which paths, which units fire), not just weights |

**Consolidation** is an optional slow outer loop. Genomes that get used a lot can occasionally be distilled into the backbone or spawned as a new expert, like sleep. This is out of scope for the semester, but it is the natural "phase 2" story.

### 1.4 Where this is *not* a good idea (honest limits)

- **On a clean, differentiable, stationary objective, gradient wins.** A linear probe trained with Adam will beat a GA-trained probe on cross-entropy. The EA has to earn its place with at least one of: non-differentiable objective, hard or discrete decisions, many small contexts, rollback / zero forgetting, or a genuine multi-modal search.
- **A frozen decoder after the EA layer costs N× its compute.** Your whiteboard's "second model at the end" is only free if it is cheap (a codebook, a small MLP) or if its input is cacheable. A 7B decoder after the EA layer destroys compute separability.
- **Frozen components cap what adaptation can reach.** If no combination of experts solves the new task, no search over combinations will. The expert library's *coverage and diversity* becomes the bottleneck. That is exactly why your "diversified experts" note matters (Idea 2).
- **Physical robots can't parallelize trials.** EA parallelism needs a parallel evaluator: cached features, a simulator, a learned model or a fleet. Idea 4 deals with this explicitly.

---

## Part 2 — Shared math formulation

### 2.1 Objects

- Input stream $x \sim \mathcal{D}_c$, where the context $c \in \mathcal{C}$ is a task, domain or damage condition, and may drift over time.
- Frozen feature map $z = \phi(x) \in \mathbb{R}^{d_z}$, and frozen experts $f_k : \mathbb{R}^{d_z} \to \mathbb{R}^{d_o}$, $k = 1..K$.
- **Genome** $g \in \mathcal{G}$, with $\dim \mathcal{G} = n$ small. It can be continuous ($\mathbb{R}^n$), discrete ($\{0,1\}^n$, $\{-1,0,1\}^n$) or mixed.
- Evolved operator $A_g$, for example routing $A_g(z) = \sum_k \pi_{g,k}(z)\, f_k(z)$, masking, or thresholded firing.
- Frozen head $\psi$: $\hat y = \psi(A_g(z))$.
- Fitness $F_c(g) = \mathbb{E}_{x\sim\mathcal{D}_c}\, R(\hat y, x)$. $R$ can be anything scalar, such as accuracy, pass@k, worst-group accuracy or episode return.

### 2.2 The cost model (why "late" matters, quantitatively)

Let the full network have $L$ layers with per-layer, per-input cost $C$, and let the EA layer sit at depth $\ell$. Take batch size $B$ and population size $N$.

**Naive population evaluation:**

$$T_{\text{naive}} = N \cdot B \cdot L \cdot C$$

**Compute-separable evaluation.** The prefix is shared and the suffix is per member:

$$T_{\text{sep}} = B\,\ell\,C \;+\; N\,B\,(L-\ell)\,C \;+\; N\,B\,C_{A}$$

**Speedup:**

$$\frac{T_{\text{naive}}}{T_{\text{sep}}} = \frac{N\,(L + C_A/C)}{\ell + N(L-\ell) + N C_A/C} \;\xrightarrow{\ \ell \to L,\ C_A \ll C\ }\; N.$$

So an EA layer placed at the very end with a cheap operator gets the **whole population for the price of one forward pass**. Every layer after the EA layer costs a factor of N.

The shape of this trade-off is the practical reason to keep the frozen decoder ψ tiny or to cache it. For the MoE case in Idea 1, EA on routers in the last $L'$ of $L$ layers costs a fraction $L'/L$ of naive per member. Prefix activations up to layer $L-L'$ are computed once.

### 2.3 Search distributions (one framework, three regimes)

Maintain a search distribution $p_\theta(g)$ and maximise $J(\theta) = \mathbb{E}_{g\sim p_\theta} F(g)$ with a natural-gradient / IGO update (Ollivier et al., 2011):

$$\theta \leftarrow \theta + \eta\, \tilde\nabla_\theta J, \qquad \tilde\nabla_\theta J \approx \frac{1}{N}\sum_{i=1}^N u_i\, \mathcal{I}(\theta)^{-1}\nabla_\theta \log p_\theta(g_i),$$

where $u_i$ are rank-based utilities. With common random numbers, which Decision 0004 makes mandatory, this covers three regimes:

| genome | $p_\theta$ | instantiation | good for $n$ up to |
|---|---|---|---|
| continuous, small | $\mathcal{N}(m, \sigma^2 C)$ | **CMA-ES** | $\sim 10^3$ (full cov), $\sim 10^5$ (sep-/LM-CMA) |
| continuous, large / matrix | $\mathcal{N}(W, \sigma^2 I)$, low-rank samples | **EGGROLL** | $10^9$, but see §1.2(a) |
| discrete | product of categoricals $p(g_j=v)=\text{softmax}(\theta_j)_v$ | **PBIL / cGA / IGO-categorical**, or a plain $(\mu+\lambda)$ GA | $10^4$–$10^6$ bits |

For categorical IGO the update reduces to a PBIL-style rule:

$$\theta_{j,v} \leftarrow \theta_{j,v} + \eta \sum_i u_i \big(\mathbb{1}[g_{i,j}=v] - p_\theta(g_j=v)\big).$$

### 2.4 Continual adaptation with a genome archive

Contexts arrive as a stream $c_1, c_2, \dots$. The system keeps an archive $\mathcal{M} = \{(\kappa_j, g_j, \hat F_j)\}$, where $\kappa_j$ is a context key such as the mean or covariance of $\phi(x)$ on context $j$, or a behavioural descriptor. On a new context $c$:

1. **Recall.** Evaluate the whole archive, or its top-$M$ by key similarity, on a probe batch. Thanks to separability this is **one parallel evaluation**. Take $g_0 = \arg\max_{g\in\mathcal{M}} \hat F_c(g)$, or initialise $p_\theta$ as a mixture over the top entries.
2. **Adapt.** Run $T$ generations of the search from §2.3 on $F_c$.
3. **Store.** Insert $(\kappa_c, g^\star_c, \hat F_c)$. Use MAP-Elites-style replacement if descriptors are used, to keep the archive diverse.

**Metrics** (standard continual learning plus adaptation speed):

- average accuracy $\bar A_T = \frac1T \sum_t A_{T,t}$;
- backward transfer $\text{BWT} = \frac{1}{T-1}\sum_{t<T}(A_{T,t} - A_{t,t})$, which is 0 by construction for exact recall and still worth reporting because retrieval can fail;
- **adaptation speed** $\text{AUC}_c = \sum_{s=1}^{S} F_c(g^{(s)})$ over the first $S$ evaluations. This is the number that shows whether memory helps.

### 2.5 A theory hook for discrete genomes (your turf)

If the genome is a bitstring and fitness is (approximately) linear, $F(g) = \sum_j w_j g_j$, then classical results apply:

- the (1+1) EA optimises linear functions in $\Theta(n \log n)$ expected time (Droste, Jansen & Wegener 2002; Witt 2013);
- more interestingly for continual learning, **re-optimisation** after a change only has to travel the Hamming distance $k$ between old and new optimum. For OneMax-like landscapes this takes $O(n\log k)$ rather than $O(n \log n)$. There are rigorous re-optimisation results for linear functions under dynamic constraints (Shi, Schirneck, Friedrich, Kötzing & Neumann, Algorithmica 2019) and for dynamic OneMax variants (Kötzing, Lissovoi & Witt 2015).

This gives a **provable "adapts fast" statement**: adaptation time scales with the *size of the change* rather than the size of the model. For a ternary readout on frozen features, the per-output-unit margin surrogate $\sum_x y_x \langle w, z_x\rangle$ is linear in $w$, so the theory applies to the surrogate exactly and to accuracy approximately. How far "approximately" goes, measured as epistasis, is itself a measurable research question.

---

## Part 3 — Literature map

| cluster | work | what it shows / how it relates |
|---|---|---|
| **Frozen features + tiny evolved controller** | Ha & Schmidhuber, *World Models*, arXiv:1803.10122 | VAE + MDN-RNN trained with gradients; **867-param controller trained with CMA-ES**. The canonical proof that this split works. |
| | Such et al., *Deep Neuroevolution*, arXiv:1712.06567; Salimans et al., arXiv:1703.03864 | Deep EAs are feasible but expensive: the "deep" end of the whiteboard question. |
| **Gradient-free composition of experts** | **LoraHub**, arXiv:2307.13269 (COLM 2024) | Few-shot cross-task generalisation by gradient-free (nevergrad) search over LoRA mixing weights; FLAN-T5 + BBH. **Closest to Idea 2.** |
| | **Transformer²**, arXiv:2501.06252 (ICLR 2025) | SVF "expert vectors" (scalings of singular values); test-time adaptation by mixing them, incl. **CEM few-shot search** over mixing coefficients. |
| | Arrow routing / LoRA libraries, arXiv:2405.11157 | Zero-shot routing among a LoRA library; the gradient-free *non-search* baseline. |
| | Task arithmetic arXiv:2212.04089; AdaMerging arXiv:2310.02575 | Merging coefficients learned with gradients (entropy minimisation); the gradient baseline for coefficient search. |
| **Evolutionary model merging** | Akiba et al., *Evolutionary Optimization of Model Merging Recipes*, arXiv:2403.13187 | CMA-ES over parameter-space and data-flow (layer-path) merges. The data-flow part is PathNet-like. |
| | **CycleQD**, arXiv:2410.14735 | Quality-diversity over merged models for agent skills. |
| | **M2N2**, *Competition and Attraction Improve Model Fusion*, arXiv:2508.16204 (GECCO 2025) | Evolves merge boundaries; competition for limited resources as the diversity mechanism. A DEGA-cousin you should read. |
| **MoE routing without gradients** | **C3PO**, arXiv:2504.07964 | Test-time re-mixing of expert weights in critical layers of pretrained MoE LLMs, using surrogates from successful reference neighbours; **+7–15% accuracy**. **The target for Idea 1.** |
| | EEP, arXiv:2407.00945 | Evolutionary search for expert pruning / merging in MoE. |
| | Aux-loss-free load balancing, arXiv:2408.15664 | Per-expert **bias** terms steer routing. Suggests a low-dim genome: evolve the biases. |
| | OLMoE, arXiv:2409.02060 | Fully open MoE (64 experts, 8 active); your current testbed. |
| **Test-time adaptation, forward passes only** | **FOA**, arXiv:2404.01650 (ICML 2024 oral) | CMA-ES on input prompt embeddings of a frozen ViT plus activation shifting; ImageNet-C; also runs on quantised models. |
| | BBT, arXiv:2201.03514; intrinsic dimension arXiv:1804.08838, arXiv:2012.13255 | Why low-dim search suffices. |
| **Modular paths / masks for continual learning** | **PathNet**, arXiv:1701.08734 | GA selects paths through module layers; freezes the paths of previous tasks. **The 2017 ancestor of this whole proposal.** |
| | **Supermasks in Superposition**, arXiv:2006.14769; edge-popup arXiv:1911.13299; supermasks arXiv:1905.01067 | Binary masks over *fixed random* weights reach good accuracy; one mask per task gives **zero forgetting at ~bits per task**. Masks are found by gradient (score + STE); nobody has seriously evolved them at scale. → Idea 3. |
| | Ternary Feature Masks, arXiv:2001.08714; Progressive Nets, arXiv:1606.04671 | More zero-forgetting-by-construction baselines. |
| | L2P arXiv:2112.08654; RanPAC arXiv:2307.02251 | Frozen pretrained ViT + small adaptive part. **RanPAC's frozen random projection + closed-form head is the baseline to beat** on frozen-feature continual learning. |
| **Discrete "neurons firing"** | WANN arXiv:1906.04358; Tsetlin machines arXiv:1804.01508; Differentiable logic gate nets arXiv:2210.08277; BitNet b1.58 arXiv:2402.17764 | Discrete / ternary computation works; evolved topology with shared weights works (WANN). |
| **Diversified expert libraries** | Branch-Train-Merge arXiv:2208.03306; Branch-Train-MiX arXiv:2403.07816 | Build experts by domain, then combine. |
| | DivDis arXiv:2202.03418; D-BAT arXiv:2202.04414; DvD arXiv:2002.00632 | Train models to **disagree where it matters**. Formalises your "agreeable vs non-agreeable questions" KL idea. |
| **Fast adaptation from a repertoire (robotics)** | Cully et al., *Robots that can adapt like animals*, arXiv:1407.3501 (Nature 2015) | MAP-Elites repertoire offline, then Bayesian optimisation over it after damage; recovers in minutes. |
| | RTE arXiv:1610.04213; APROL arXiv:1907.07029; Black-DROPS arXiv:1703.07261 | Repertoire adaptation with learned models; Black-DROPS runs **CMA-ES inside a learned model**. |
| | QDax arXiv:2202.01258 / arXiv:2308.03665 | Massively parallel QD on GPU (Brax). Infrastructure for Idea 4. |
| **Evolved plasticity** | Najarro & Risi, arXiv:2007.02686 | Evolve Hebbian rules, not weights; random init adapts within an episode; robust to morphology damage. → Idea 5. |
| | CAVIA arXiv:1810.03642; ES-MAML arXiv:1910.01215 | Adapt only a small context vector; ES as the meta-learner. |
| **Your substrate** | EGGROLL arXiv:2511.16652; ES at scale arXiv:2509.24372; ES forgetting arXiv:2601.20861 | Already in `research/literature/`. |

**Gap statement.** Most of the works above do one of three things: (i) evolve a small late thing **once**, as a one-shot merge or recipe; (ii) adapt at test time **without memory**, as FOA, C3PO and Transformer² do; or (iii) do continual learning with **gradient-found** masks or prompts, like SupSup and L2P. The only one close to the full loop is **PathNet** (GA + frozen modules + continual), and it predates pretrained backbones and GPU-parallel EAs.

The Evolve contribution: **a pretrained or diversified library, a late separable EA layer that exploits massive parallelism, and a genome archive as memory, evaluated on adaptation speed and forgetting.**

---

## Part 4 — The ideas

Each idea follows the same template: **idea → why an EA → math → novelty vs literature → benchmark (concrete, start-tomorrow) → kill criterion.**

---

### Idea 1 — EvoRouter: evolved late-layer routing for pretrained MoE, with a routing-genome memory

**Idea.** Take a pretrained MoE (OLMoE-1B-7B). Freeze everything. Evolve a **tiny routing genome** that acts only on the last $L'$ MoE layers: per-expert bias shifts, optionally conditioned on a compressed prompt embedding. Each domain or task gets its own genome, stored in an archive. New domains start from the best archived genome. This is Direction C from `E-new-directions.md` scaled down to where ES is strong.

**Why an EA.** Routing is top-k: a hard, discrete, non-differentiable choice. Gradient methods must relax it (soft routing), which causes the train/inference mismatch your toy experiment exposed: 23.8% for gradient vs 50.5% for ES. C3PO avoids gradients but optimises **surrogates** built from neighbours. ES can optimise the **actual hard-routed task metric**.

**Math.** At MoE layer $\ell$, the router logits are $s_\ell(h) = W_\ell h$, with $h \in \mathbb{R}^{H}$, $W_\ell \in \mathbb{R}^{K\times H}$, $K = 64$ and top-8 selection. The genome modifies logits in the last $L'$ layers:

$$\tilde s_\ell(h) = W_\ell h + b_\ell + U_\ell\, P\,\bar e(x), \qquad \ell \in \{L-L'+1,\dots,L\}$$

- $b_\ell \in \mathbb{R}^K$ is a per-expert bias shift, the same knob as DeepSeek-V3's aux-loss-free balancing.
- $\bar e(x)$ is the mean-pooled prompt embedding and $P \in \mathbb{R}^{d'\times H}$ is a **fixed** random projection, $d' = 8$–$32$.
- $U_\ell \in \mathbb{R}^{K\times d'}$ is an input-conditioned correction, optional in stage 2.

Genome sizes:

- bias-only: $n = K L' = 256$ for $L'=4$, which is ideal for CMA-ES;
- with conditioning: $n = K L'(1+d') \approx 4\text{–}8\text{k}$, which suits sep-CMA or EGGROLL.

Fitness on context $c$ uses a small reference set $\mathcal{R}_c$ (e.g. 32–128 labelled examples), scored as multiple-choice log-likelihood or accuracy:

$$F_c(g) = \frac{1}{|\mathcal{R}_c|} \sum_{(x,y)\in\mathcal{R}_c} \Big[\log p_g(y\mid x) \;\text{or}\; \mathbb{1}[\arg\max_{y'} p_g(y'\mid x) = y]\Big]$$

To prevent overfitting a 64-example set, use a KL leash:

$$F_c(g) - \beta\,\mathbb{E}_x\, \mathrm{KL}\big(\pi_g(\cdot\mid x)\,\|\,\pi_0(\cdot\mid x)\big),$$

where $\pi$ denotes routing distributions. This penalty is computable for free because base routing is computed in the shared prefix anyway.

**Separability.** Multiple-choice scoring is a single forward pass on a fixed input, so compute separability holds for layers $< L-L'+1$. Compute and cache hidden states up to layer $L-L'$ once, then evaluate N genomes on the last $L'$ layers only. For OLMoE with 16 layers and $L'=4$, that is **~4× cheaper per member than naive**, before any other trick.

**Continual variant.** Stream the domains (e.g. the six C3PO benchmarks, then MMLU subject groups as further contexts). Key = mean of $\bar e(x)$ over the reference set. Use the archive recall from §2.4.

**Novelty vs literature.**

- **vs C3PO:** real-objective population search instead of neighbour-surrogates, plus a memory / archive and continual evaluation.
- **vs EEP:** routing adaptation rather than pruning.
- **vs your toy Direction C:** pretrained experts, a realistic scale, and an EA restricted to a low-dimensional late genome.

**Benchmark (start now).**

- **Model:** OLMoE-1B-7B-0924 (bf16 on a 24 GB 3090/4090, or the 4-bit path already in `run_olmoe.py` for a T4).
- **Tasks:** follow C3PO's protocol and its six benchmarks, so their numbers are directly comparable. Reference set: the training split; evaluation: the test split.
- **Arms** (all at a matched number of forward passes):
  1. base model;
  2. ICL (5-shot);
  3. **gradient on the same genome** (soft-routing relaxation of $b_\ell$, Adam) — the key control;
  4. random search on the same genome, as a noise control (rule 4);
  5. CMA-ES bias-only;
  6. CMA-ES / EGGROLL with conditioning;
  7. C3PO (reported numbers, plus their code if runnable);
  8. archive-recall only (no adaptation);
  9. recall + adapt.
- **Metrics:**
  - accuracy per task;
  - forward passes to reach 90% of final;
  - off-task degradation when one genome is used globally vs per-domain genomes;
  - expert load entropy;
  - for the stream: $\bar A_T$, BWT and $\text{AUC}_c$.
- **Compute:** a 256-dim CMA-ES with N=32 over 100 generations on 64 references is about 200k single-sequence forward passes of 4 layers. That is hours, not days, on one 4090.
- **Two-week plan:**
  - week 1: prefix caching plus bias-only CMA-ES on one task, sanity-checked against the gradient arm;
  - week 2: all six tasks plus the stream.

**Kill criterion.** If CMA-ES bias-only is not ≥ the gradient arm on the same genome at matched compute on at least 4 of 6 tasks, the "hard routing needs ES" argument does not survive real pretrained experts. In that case, fold routing into Idea 2 as one knob among several.

---

### Idea 2 — EvoCompose: evolutionary composition over a *diversified* expert library, with a genome archive

**Idea.** This is the whiteboard's use-case list, items 2–5, made concrete.

1. **Build a library** of $K$ small experts: LoRAs or SVF vectors on a frozen base. Train them **to be diverse**: each is good at its domain and they deliberately disagree where the ensemble is uncertain. This is your "discounted KL / agreeable vs non-agreeable" idea.
2. **Compose them** with a tiny genome of mixing coefficients, per layer group, **only in the last layers** so that composition is compute-separable. The genome is searched by an EA per task or context.
3. **Remember** the composition genomes in an archive and recall them for new tasks.

This is the flagship because it tests the *whole* Evolve thesis: diverse slow components, a fast evolved combination, and memory.

**Why an EA.** The mixing landscape is **multi-modal and flat-then-cliff**: many coefficients do nothing and some combinations interfere. LoraHub and Transformer² already found gradient-free search works here with 5–20 labelled examples. Population search also gives you diversity for free: the final population is a set of *different* good compositions, which is a natural pass@k ensemble.

**Math — library construction (the diversity objective).** Train experts $\{\Delta_k\}_{k=1}^K$ on domains $\{\mathcal{D}_k\}$. Let $p_k(\cdot\mid x)$ be expert $k$'s predictive distribution and $a(x) \in [0,1]$ an **agreeableness** weight: high when the question has a single correct answer that experts should agree on, low when it is genuinely ambiguous or out-of-domain.

$$\mathcal{L}_k = \underbrace{\mathbb{E}_{x\sim\mathcal{D}_k} \ell_{\text{CE}}(p_k, y)}_{\text{expertise}} \;-\; \lambda\, \mathbb{E}_{x\sim\mathcal{D}_{\text{pool}}}\Big[(1-a(x)) \cdot \frac{1}{K-1}\sum_{j\neq k} \mathrm{JS}\big(p_k(\cdot\mid x)\,\|\,\mathrm{sg}[p_j(\cdot\mid x)]\big)\Big]$$

- JS rather than KL for boundedness; $\mathrm{sg}$ is stop-gradient (train the experts round-robin).
- A practical choice is $a(x) = \max_j \big(p_j(y^\star\mid x)\big)$ when a label exists: *if someone already knows the answer, don't push them apart*. On unlabelled pool data use $a(x) = 1 - \bar H(x)/\log|\mathcal{Y}|$ (DivDis-style).
- An expertness weighting of the repulsion, $w_{jk}(x) \propto \text{conf}_k(x)\,\text{conf}_j(x)$, gives your "balance KL by expertness".

**Math — composition genome.** Let the last $L'$ layers be split into $G$ groups. Genome $g = \{\alpha_{\gamma,k}\}$ with $\gamma \le G$, $k \le K$, so $n = GK$ (e.g. $G = 4$, $K = 16$ gives $n = 64$). For LoRA experts $\Delta_k = B_k A_k$ at a layer in group $\gamma$:

$$W' = W + \sum_{k=1}^K \alpha_{\gamma,k}\, B_k A_k, \qquad y = W'u = Wu + \sum_k \alpha_{\gamma,k}\, B_k (A_k u).$$

**Parallel trick.** $A_k u$ (rank $r$) is shared across all members that see the same $u$. Each member only pays $\sum_k \alpha_{\gamma,k}B_k(\cdot)$, so the per-member overhead is $O(K r (m+n))$ per token. This is EGGROLL's economics again.

Fitness is the few-shot objective as in Idea 1, with an L1 or KL leash toward $\alpha = 0$ (LoraHub uses L1).

**Search.** CMA-ES for $n \le 10^3$. Initialise from the archive (§2.4). Optionally use **partitioned perturbation over layer groups** (Decision 0006): each member perturbs one group's coefficients only. E10 predicts this helps the weak-gradient groups, and it is directly testable here.

**Novelty vs literature.**

- **vs LoraHub / Transformer²:** (i) a **trained-for-diversity** library, with the ablation *diverse vs independently trained experts*, which nobody has run; (ii) late-only composition for compute separability, so populations of thousands become feasible; (iii) archive recall over a task **stream**.
- **vs Sakana merging / CycleQD / M2N2:** those are offline, one-shot merges. This is online, per-context and memory-backed.

**Benchmark (start now) — two stages.**

**Stage A (no training needed, published baselines).** LoraHub's setup.

- Base: FLAN-T5-large (~780M); the ~200 LoRA modules released by the LoraHub authors on Hugging Face.
- **BBH (27 tasks), 5 labelled examples per task as fitness.**
- Arms:
  1. zero-shot;
  2. ICL;
  3. LoraHub (their code);
  4. CMA-ES same space;
  5. late-only composition with $N \in \{32, 256, 2048\}$;
  6. per-group partitioned;
  7. random-search control;
  8. **stream**: BBH tasks in a fixed random order, with archive recall + adapt vs from scratch.
- Metrics:
  - BBH exact-match accuracy per task;
  - evaluations-to-X;
  - $\text{AUC}_c$ over the stream;
  - variance across seeds (LoraHub reports high variance, and ES is known to reduce it: see arXiv:2509.24372's 15× lower std).
- Runs on one 3090/4090 or even the 1080 Tis.

**Stage B (tests the diversity hypothesis).**

- Base: Qwen2.5-0.5B-Instruct.
- $K = 8$–$16$ LoRAs (rank 8) trained on distinct domains: e.g. Countdown variants from your E2 code, GSM8K-style arithmetic, code, logic grids, and a few FLAN clusters.
- Two libraries of equal compute: **independent** vs **diversity-regularised** (the $\mathcal{L}_k$ above, $\lambda \in \{0.1, 0.3\}$).
- Held-out composite tasks. Compare composition performance and pass@k.
- Required control (rule 2): a λ sweep, plus a library trained with *random* repulsion targets (rule 4).

**Kill criterion.**

- Stage A: if late-only composition loses more than ~1 point vs all-layer composition, compute separability costs too much; fall back to all-layer composition with EGGROLL-style batching.
- Stage B: if the diversity-regularised library does not beat the independent one on held-out composition at matched compute, the "diversify experts" part is dead and the idea reduces to a stronger LoraHub.

---

### Idea 3 — The Sandwich: frozen encoder → discrete evolved "firing" layer → frozen decoder

**Idea.** This is the whiteboard's boldest part, which I don't think is bullshit, but it needs the right benchmark.

- A frozen pretrained encoder produces $z$.
- An **evolved ternary layer** produces binary "spikes" $h = \mathbb{1}[Wz > \tau]$, with $W \in \{-1,0,1\}^{m\times d}$.
- A **frozen decoder** maps $h$ to the output.

The decoder is fixed from the start, e.g. an error-correcting output code (ECOC) codebook, so **new classes or tasks don't retrain the decoder**. They only need new codewords and an adapted genome.

Continual learning happens by **growing units**. Each new task adds $m_t$ new units, and the EA searches only their bits. Old units are frozen, so there is zero interference. The genome is literally a bitstring, and the theory from §2.5 applies.

**Why an EA.** Ternary weights and thresholded firing are non-differentiable; gradient methods need straight-through estimators, which are biased. GAs handle bitstrings natively. Fitness evaluation on cached features is a single matmul, so evaluating **10⁵ genomes per second on one GPU** is realistic: this is the "swarm computing" bullet on your whiteboard. And it is the only idea where **rigorous runtime analysis** is within reach.

**Math.**

- Cached features: $Z \in \mathbb{R}^{B\times d}$ from a frozen DINOv2 / CLIP ViT. Use a fixed random ±1 projection first if $d$ is too large (RanPAC-style), with $d \sim 1\text{–}2$k.
- Units: $h = \mathbb{1}[Z W^\top > \tau] \in \{0,1\}^{B \times m}$. Genome $g = \mathrm{vec}(W) \in \{-1,0,1\}^{md}$, optionally with per-unit $\tau$.
- Frozen decoder: class codebook $Q \in \{0,1\}^{C \times m}$ with random rows at large pairwise Hamming distance. Prediction: $\hat y = \arg\min_c \mathrm{Ham}(h, Q_c)$.
- Fitness is accuracy or any non-differentiable metric, e.g. worst-class or worst-group accuracy:

$$F(g) = \frac{1}{B}\sum_{b} \mathbb{1}[\hat y_b = y_b] \quad\text{or}\quad \min_{\text{group } s} \text{acc}_s(g).$$

**Separability within the genome.** Given the codebook, each unit $u$'s *target* bit for example $b$ is known: $Q_{y_b,u}$. The per-unit fitness

$$F_u(w_u) = \sum_b \mathbb{1}\big[\mathbb{1}[\langle w_u, z_b\rangle > \tau_u] = Q_{y_b,u}\big]$$

decomposes over units. The EA therefore runs **$m$ independent small searches in parallel**, and the global accuracy only couples them through the Hamming decoder. This is the perceptron-per-neuron structure you drew. The margin surrogate $\sum_b (2Q_{y_b,u}-1)\langle w_u, z_b\rangle$ is **linear** in $w_u$, which is the OneMax/linear-function regime of §2.5.

**Continual.** Task $t$ brings classes $\mathcal{Y}_t$ and adds units $m_t$ plus codewords. Old codewords are extended with 0s on the new units, or the new units are left unconstrained for old classes. Only $W_{\text{new}}$ is searched, and memory is $m_t \cdot d \cdot \log_2 3$ bits per task.

**Research questions.**

- (i) GA vs STE-gradient on the same ternary layer: accuracy and wall-clock.
- (ii) Measured epistasis: how far accuracy is from linear in bits.
- (iii) Re-optimisation time vs Hamming shift under distribution drift, compared with the $O(n\log k)$ prediction.
- (iv) Zero-forgetting continual accuracy vs RanPAC.

**Novelty.** SupSup and Ternary Feature Masks find masks by gradient on randomly initialised or task-trained nets. Tsetlin machines and WANNs evolve discrete structure but not on top of pretrained features in a continual setting. **A GA-found ternary layer over frozen foundation-model features, with a fixed ECOC decoder, unit-growth continual learning and runtime analysis** is, as far as I can find, open.

**Benchmark (start now).**

- **Primary:** class-incremental **Split CIFAR-100** (10 tasks × 10 classes) and **ImageNet-R** (10 tasks) on frozen ViT-B/16 (IN-21k) features. This is the exact setting of L2P / RanPAC, so the baselines are published.
  - Extract features once. Everything after that is a few KB of matmul per evaluation, so this runs on a MacBook or a 1080 Ti.
  - Arms:
    1. linear probe (Adam, joint) as upper reference;
    2. **RanPAC**;
    3. STE-trained ternary layer (same architecture);
    4. $(\mu+\lambda)$ GA / cGA ternary layer;
    5. PBIL;
    6. GA + archive recall under a **drift** protocol: re-present old tasks with corrupted features (e.g. ImageNet-C style corruptions of the images) and measure re-adaptation time.
  - Metrics:
    - final average accuracy $\bar A_T$ and BWT;
    - bits per task;
    - evaluations and wall-clock to convergence;
    - re-adaptation time vs measured Hamming shift.
- **Secondary (shows off the non-differentiable objective):** Waterbirds / CelebA **worst-group accuracy** on frozen features, optimised *directly* by the GA on the validation set. The strong baseline is **DFR** (Kirichenko et al., arXiv:2204.02937, last-layer retraining on group-balanced data). Compute is the same and cheap.

**Kill criterion.**

- If the GA ternary layer lands more than ~5 points below the STE ternary layer on Split CIFAR-100 at matched wall-clock, the discrete-EA angle is not competitive for accuracy. It would then survive only as a theory paper on re-optimisation, which may still be worthwhile given your background.
- If it matches STE, you have a clean empirical-plus-theory story.

---

### Idea 4 — Evolved adaptation over a QD repertoire (robotics), with the EA running inside a learned model

**Idea.** The whiteboard's "Robotics" and "planning" bullets.

- **Offline:** build a diverse repertoire of low-level skills or policies with QD (MAP-Elites / DCG-ME in QDax). These are the "diversified experts".
- **Online, after damage or environment change:** a tiny genome selects and blends skills. Examples: a gait parameter vector over the repertoire's descriptor space, or mixing weights over the top-$M$ policies.
- **Evaluation:** massively parallel **inside a learned residual dynamics model**, with only a few real trials used to update that model.

This solves the "real robots can't parallelise" problem of §1.4: the EA is parallel where evaluation is cheap (model, GPU) and frugal where it is expensive (real trials).

**Why an EA.** Evaluation inside a learned model is noisy, non-smooth and multi-modal. Black-DROPS showed CMA-ES in a learned model is data-efficient. Brax/MJX now makes $10^4$ parallel model rollouts trivial.

**Math.**

- Repertoire $\mathcal{P} = \{(\pi_j, d_j, f_j)\}$ with descriptors $d_j$ (e.g. foot-contact pattern) and prior fitness $f_j$.
- Genome $g = (\text{descriptor target } d^\star, \text{blend temperature } T, \text{residual gains } k)$ with $n \approx 10$–$50$. The policy is

$$\pi_g(s) = \sum_{j \in \text{kNN}(d^\star)} w_j(g)\, \pi_j(s) + k^\top \phi_{\text{ctrl}}(s).$$

- Learned model: $\hat s_{t+1} = s_{t+1}^{\text{sim}} + \delta_\omega(s_t,a_t)$, with residual $\delta_\omega$ fitted on the real (here: "damaged sim") transitions collected so far.
- Adaptation loop per real trial $t$: run CMA-ES on $\hat F_\omega(g)$ with $N \times$ model rollouts (thousands), execute the best $g$ once in the real env, add the data, refit $\delta_\omega$.
- Archive: store $g^\star$ per damage condition, keyed by the residual model's parameters or a trajectory embedding.

**Benchmark (start now).**

- **Env:** Brax/MJX Ant or QDax Hexapod; **damage** = disable or weaken one or two legs' actuators, following the protocol of Cully et al. 2015.
- Repertoire from QDax MAP-Elites (hours on one GPU).
- Arms:
  1. ITE (GP-BO over repertoire; reimplemented, since it is simple);
  2. RTE / APROL if time;
  3. PPO fine-tune from the best repertoire policy;
  4. domain randomisation;
  5. **ours:** CMA-ES in the learned model;
  6. ours + archive under a *sequence* of damages (continual).
- **Metric:** real trials to recover X% of undamaged performance; final performance; total wall-clock.

**Kill criterion.** If ITE (sequential BO, no model) matches ours on trials-to-recovery, the parallel-EA-in-model machinery isn't buying anything. That is quite possible, since ITE is very strong in low dimensions. This is the riskiest idea in terms of engineering time; I'd do it only if robotics is where you want to end up.

---

### Idea 5 (stretch) — Evolve the *learning rule* of the late layer, not its weights

**Idea.** Instead of evolving the genome per context, evolve **once** the parameters of a local plasticity rule for the late layer. The layer then adapts *itself* online, within an episode or context, with no outer search at deployment. This is Najarro & Risi's result on top of a frozen backbone.

**Math.** For late-layer weights $w_{ij}$ between pre-activation $o_j$ (from frozen features) and post-activation $o_i$:

$$\Delta w_{ij} = \eta_{ij}\,\big(A_{ij}\, o_i o_j + B_{ij}\, o_i + C_{ij}\, o_j + D_{ij}\big) \cdot m(t),$$

where $m(t)$ is a neuromodulatory signal such as the reward-prediction error. The genome is $\{A,B,C,D,\eta\}$, usually shared per unit type to keep $n$ small. The outer ES optimises $\mathbb{E}_{c\sim\mathcal{C}}\left[\sum_t R_t\right]$ over a *distribution of contexts*, which is meta-learning by ES.

**Why here.** This is the most "self-adapting" option. The EA's job moves from solving tasks to producing a learner. Adaptation speed at deployment is a forward pass plus a local update: no population, no memory. It pairs with Idea 3's discrete units (plastic thresholds $\tau$) or Idea 4's residual gains.

**Benchmark.** The Idea 4 damaged Ant, where Najarro & Risi reported robustness, *or*, much cheaper, the Idea 3 feature stream with drifting corruptions. The comparison is plastic late layer vs static evolved layer vs static + re-search.

---

## Part 5 — Comparison and recommendation

| | Idea 1 EvoRouter | Idea 2 EvoCompose | Idea 3 Sandwich | Idea 4 Repertoire | Idea 5 Plasticity |
|---|---|---|---|---|---|
| Time to first result | **~1 week** (code + Colab exist) | ~1 week (Stage A) | **~3 days** (cached features) | 3–4 weeks | 3+ weeks |
| Compute | 1× 4090 / T4-4bit | 1× 3090 (A), 1× 4090 (B) | MacBook / 1080 Ti | 1 GPU (JAX) | varies |
| Published baseline to beat | **C3PO** | **LoraHub**, Transformer² | **RanPAC**, L2P, DFR | ITE, RTE, APROL | Najarro & Risi |
| EA-specific advantage | hard top-k routing | multi-modal mixing, population = ensemble | discrete bits, theory, 10⁵ evals/s | parallel model rollouts | meta-level |
| Continual / memory story | routing-genome archive | composition archive, **full Evolve loop** | unit growth, zero forgetting | damage archive | learner, not memory |
| Fit with existing repo | **high** (Direction C, OLMoE) | high (EGGROLL, partitioned, countdown) | medium (new, but tiny) | low | low |
| Novelty risk | medium (C3PO is close) | medium (LoraHub is close; diversity part new) | **low** (open gap + theory) | medium | medium |
| Paper-ability | workshop | **main-track if Stage B works** | **theory + empirical, GECCO/FOGA-shaped** | robotics venue | workshop |

**Suggested path:**

1. **Weeks 1–2:** run Idea 1 bias-only and Idea 3 primary in parallel. Both are cheap, and both answer the same core question: *does a late, low-dim evolved layer beat a gradient-trained layer of the same shape when the decision is hard or discrete?*
2. **Pick by result.** If Idea 1 wins, go into Idea 2, which is the same machinery plus library construction and memory. If Idea 3 wins, it becomes the theory-heavy thesis direction.
3. Keep Ideas 4 and 5 as extensions, not starting points.

## Part 6 — Standing rules that apply to every idea

The repo earned these the hard way; all of them bite here:

1. **Beat a tuned baseline on the same parameterisation.** The gradient arm must optimise *the same genome* (same biases, same coefficients, same ternary layer via STE). Otherwise you compare parameterisations, not optimisers.
2. **Controls must be free to vary.** Random-search controls get their own seeds per run (the E4/0005 lesson).
3. **The benchmark must be able to exhibit the failure.** For "memory helps adaptation", the stream must contain *recurring or related* contexts. A stream of unrelated tasks cannot show a recall benefit.
4. **Watch sampling noise at small σ (E12).** Fitness on 5–64 references is noisy. Use common random numbers and fixed reference batches per generation, and check the reliability of the fitness ranking (split-half) before trusting any comparison.
5. **Few-shot fitness overfits.** Always report on held-out data, and keep the KL/L1 leash as a swept hyperparameter, not a fixed one.

---

## References (arXiv IDs)

- **Substrate:** EGGROLL 2511.16652 · ES at scale 2509.24372 · ES forgetting 2601.20861 · Salimans 1703.03864 · Deep Neuroevolution 1712.06567 · CMA-ES tutorial 1604.00772 · IGO 1106.3708 · NES 1106.4487
- **Frozen + evolved controller:** World Models 1803.10122 · PathNet 1701.08734
- **Composition / merging:** LoraHub 2307.13269 · Transformer² 2501.06252 · Arrow/LoRA libraries 2405.11157 · Task arithmetic 2212.04089 · AdaMerging 2310.02575 · Evo model merge 2403.13187 · CycleQD 2410.14735 · M2N2 2508.16204 · BTM 2208.03306 · BTX 2403.07816
- **MoE:** C3PO 2504.07964 · EEP 2407.00945 · aux-loss-free balancing 2408.15664 · OLMoE 2409.02060
- **Test-time / low-dim:** FOA 2404.01650 · BBT 2201.03514 · intrinsic dim 1804.08838, 2012.13255
- **Continual / masks / frozen features:** SupSup 2006.14769 · edge-popup 1911.13299 · supermasks 1905.01067 · Ternary Feature Masks 2001.08714 · Progressive Nets 1606.04671 · L2P 2112.08654 · RanPAC 2307.02251 · DFR 2204.02937
- **Discrete computation:** WANN 1906.04358 · Tsetlin 1804.01508 · Logic gate nets 2210.08277 · BitNet b1.58 2402.17764
- **Diversity:** DivDis 2202.03418 · D-BAT 2202.04414 · DvD 2002.00632
- **Robotics / QD:** Cully et al. 1407.3501 · RTE 1610.04213 · APROL 1907.07029 · Black-DROPS 1703.07261 · QDax 2202.01258, 2308.03665
- **Plasticity / meta:** Najarro & Risi 2007.02686 · CAVIA 1810.03642 · ES-MAML 1910.01215
- **Theory (not arXiv):** Droste, Jansen & Wegener, TCS 2002 (linear functions) · Witt, CPC 2013 (tight bounds, linear functions) · Kötzing, Lissovoi & Witt, FOGA 2015 (dynamic OneMax) · Shi, Schirneck, Friedrich, Kötzing & Neumann, Algorithmica 2019 (re-optimisation, linear functions under dynamic constraints)

*Citation caveat:* the arXiv IDs for C3PO, FOA, Transformer², LoraHub and M2N2 were checked this session. The others are from memory and should be spot-checked before they go into a paper.
