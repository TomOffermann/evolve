# F3 — The Sandwich: Frozen Encoder → Evolved Discrete "Firing" Layer → Frozen Decoder

**Part of:** F — The Late Evolutionary Layer (`research/base/base.tex` has the shared notation and the general theory)
**Status:** Proposal · no results yet · **Date:** 2026-09-21
**One-line pitch:** *Put a layer of ternary-weighted threshold neurons between a frozen foundation-model encoder and a fixed decoder. Evolve its bits with a GA, grow new neurons for new tasks so old ones never change, and use EA runtime theory to say how fast it can re-adapt.*

This is the idea closest to your whiteboard's *"GA layer could then maybe even be discrete (binary/ternary weights) and act more in the classical perceptron 'neurons' firing idea."* It is not bullshit, but it only pays off on the right benchmark, and it is the one idea where **theory of EAs**, your home turf, has something to say.

---

## 0. Reading guide

- §1 background: frozen foundation features, ternary / threshold units, error-correcting output codes, and discrete EAs;
- §2 the idea;
- §3 the math: model, fitness, decomposition, continual growth;
- §4 the search algorithms and a GPU implementation sketch;
- §5 theory: what can and cannot be proven, including the honest caveat that the linear surrogate is trivially solvable;
- §6 prior work explained;
- §7 the benchmark protocol (primary: class-incremental learning on frozen ViT features; secondary: direct worst-group optimisation);
- §8 hypotheses, risks and kill criteria;
- §9 extensions.

---

## 1. Background

### 1.1 Frozen foundation-model features are already very good

A vision transformer pretrained on a large dataset turns an image into a feature vector $z = \phi(x) \in \mathbb{R}^{d}$. Examples: ViT-B/16 on ImageNet-21k, where $z$ is the 768-d CLS token; DINOv2; CLIP.

On these features, a *nearest class mean* classifier (average the features of each class; predict the closest mean) already reaches high accuracy on CIFAR-100 and similar benchmarks. This finding drives a whole line of continual-learning work: *SimpleCIL / ADAM* (arXiv:2303.07338), RanPAC (arXiv:2307.02251) and L2P (arXiv:2112.08654).

**Consequence.** Once features are computed once and cached, everything after the encoder is a tiny computation. A matrix $Z \in \mathbb{R}^{B \times d}$ of $B = 50{,}000$ CIFAR images × 768 floats is 150 MB. Evaluating a candidate classifier on all of it is one matmul. **This is the regime where an EA can afford $10^4$–$10^6$ fitness evaluations per second.**

### 1.2 Ternary threshold units

A classical perceptron / McCulloch–Pitts neuron fires if a weighted sum exceeds a threshold. With **ternary weights**, each input feature either excites, is ignored, or inhibits:

$$h_u(z) = \mathbb{1}\big[\langle w_u, z\rangle > \tau_u\big], \qquad w_u \in \{-1, 0, +1\}^{d}.$$

**Why ternary?**

- **Discrete genome:** a unit is a string over a 3-letter alphabet, which is GA-native and theory-friendly.
- **Sparsity:** zeros select features, so each unit becomes a readable "feature detector": fires if features $\{3, 17, 90\}$ are high and $\{5, 44\}$ are low.
- **Storage:** $\log_2 3 \approx 1.58$ bits per weight (cf. BitNet b1.58, arXiv:2402.17764, which trains ternary LLMs).
- **Gradient-hostility:** the step function and discrete weights have no useful gradient; gradient training needs a *straight-through estimator* (STE), which pretends the gradient of the step is 1. It works, but it is biased. **This is where the GA should win or at least tie.**

### 1.3 Error-correcting output codes (ECOC): a decoder that never needs training

To classify into $C$ classes with $m$ binary units, assign each class a **codeword** $Q_c \in \{0,1\}^m$ and predict the class whose codeword is nearest in Hamming distance:

$$\hat y(z) = \arg\min_{c} \mathrm{Ham}\big(h(z), Q_c\big).$$

With random codewords and $m \gg \log_2 C$, codewords are far apart, so a few wrong units can be *corrected*: if the minimum pairwise distance is $\delta$, up to $\lfloor(\delta-1)/2\rfloor$ unit errors are tolerated. The codebook $Q$ is the **frozen decoder** of the sandwich. It is fixed before any learning, and new classes only need new codewords, not a retrained decoder.

### 1.4 Discrete evolutionary algorithms, the minimal set

- **(1+1) EA:** keep one solution; flip each position with probability $1/n$ (for ternary: resample it to one of the two other values); accept if not worse.
- **$(\mu+\lambda)$ GA:** keep $\mu$ parents, create $\lambda$ offspring by uniform crossover + mutation, keep the best $\mu$ of all.
- **cGA / PBIL (estimation-of-distribution):** keep a probability vector $p_{j,v} = P(w_j = v)$ and sample a population from it. Shift $p$ towards the winners: $p_{j,v} \leftarrow p_{j,v} + \eta(\mathbb{1}[w^{\text{best}}_j = v] - p_{j,v})$. This is the discrete counterpart of ES/CMA-ES and is trivially parallel on a GPU.

---

## 2. The idea

```
   image x
     │
     ▼
 ┌──────────────────────┐   z ∈ ℝ^d  (computed ONCE, cached)
 │  frozen ViT encoder φ │──────────────────────────┐
 └──────────────────────┘                           │
                                                    ▼
               ┌──────────────── evolved layer (the genome) ─────────────────┐
               │  m ternary threshold units:  h_u = 1[⟨w_u, z⟩ > τ_u]         │
               │  w_u ∈ {−1,0,+1}^d   ← GA / PBIL, 10^4–10^6 evals/s          │
               │  task 1 units │ task 2 units │ … │ task t units (new, searched) │
               │   (frozen)    │   (frozen)   │   │                            │
               └──────────────────────────────┬──────────────────────────────┘
                                              │ h ∈ {0,1}^m  ("which neurons fire")
                                              ▼
                               ┌────────────────────────────┐
                               │ frozen decoder: codebook Q  │   ŷ = argmin_c Ham(h, Q_c)
                               └────────────────────────────┘
```

- **Encoder:** frozen and pretrained, run once per image.
- **Middle:** evolved discrete neurons. This is the only thing that learns.
- **Decoder:** frozen, defined by a codebook (ECOC).
- **Continual learning:** each new task *adds* units. Old units are frozen, so old knowledge is untouched, and the search space per task is only the new units.
- **Research value:** (i) does a GA-found discrete layer compete with gradient/STE and with strong frozen-feature baselines? (ii) can we *prove* anything about re-adaptation speed?

---

## 3. Math

### 3.1 Model

- Features: $z = \phi(x) \in \mathbb{R}^d$ (e.g. $d = 768$). Standardise them per dimension using statistics of the first task only, so later tasks don't leak.
- **Optional random expansion**, as RanPAC does: $\tilde z = \mathrm{ReLU}(R z)$, $R \in \mathbb{R}^{M\times d}$ fixed Gaussian, $M \in \{2048, 4096\}$. A random nonlinear expansion makes classes more linearly separable. Below, $z$ denotes whichever is used, with dimension $d$.
- Units $u = 1..m$: $h_u(z) = \mathbb{1}[\langle w_u, z\rangle > \tau_u]$, with genome $w_u \in \{-1,0,1\}^d$ and threshold $\tau_u \in \mathbb{R}$. $\tau_u$ can be set in closed form given $w_u$ (§3.3).
- Decoder: codebook $Q \in \{0,1\}^{C\times m}$ and $\hat y(z) = \arg\min_c \mathrm{Ham}(h(z), Q_c)$. Break ties by a soft score, e.g. the sum of margins $\langle w_u, z\rangle - \tau_u$ agreeing with $Q_c$.

### 3.2 Fitness

Let the data be $\mathcal{T} = \{(z_b, y_b)\}_{b=1}^B$. Primary fitness is accuracy:

$$F(W, \tau) = \frac1B \sum_{b} \mathbb{1}\big[\hat y(z_b) = y_b\big].$$

Any other scalar works, and this is the EA's advantage: worst-class accuracy $\min_c \text{acc}_c$, worst-group accuracy (§7.2), macro-F1, or accuracy minus a sparsity cost $\lambda\,\|W\|_0/(md)$.

### 3.3 Decomposition into per-unit problems

For fixed codewords, unit $u$ has a **target bit** for every example: $t_{b,u} = Q_{y_b, u}$. Define the **per-unit fitness** as the fraction of examples on which the unit fires as its codeword says:

$$F_u(w_u, \tau_u) = \frac1B\sum_b \mathbb{1}\Big[\mathbb{1}\big[\langle w_u, z_b\rangle > \tau_u\big] = t_{b,u}\Big].$$

**Why this is the right sub-problem.** If every unit achieves per-unit accuracy $\ge 1 - \epsilon$ and errors were independent, the number of wrong bits for an example is $\text{Bin}(m, \epsilon)$. ECOC decoding is correct whenever fewer than $\delta/2$ bits are wrong, so

$$P(\text{error}) \le P\big(\text{Bin}(m,\epsilon) \ge \delta/2\big) \le \exp\!\big(-m\,\mathrm{KL}(\tfrac{\delta}{2m}\,\|\,\epsilon)\big)$$

(Chernoff), which is tiny when $\epsilon < \delta/(2m)$. For random codes $\delta/m \approx 1/2$, so units only need to beat $\epsilon < 1/4$. **Weak units combine into a strong classifier; this is the boosting intuition behind ECOC.** Errors are not independent in practice, so the bound is optimistic, but the decomposition still gives:

- **$m$ independent searches:** embarrassingly parallel, no coordination;
- **optimal threshold in closed form:** for fixed $w_u$, sort the projections $\langle w_u, z_b\rangle$ and choose $\tau_u$ to maximise $F_u$ by a single sweep ($O(B\log B)$). The genome is therefore only $w_u$.

A **global refinement** phase afterwards optimises the true $F$ (all units jointly) for a few generations. This handles correlated errors.

### 3.4 Continual learning by unit growth

Tasks $t = 1..T$ arrive, each with a new set of classes $\mathcal{Y}_t$. **Class-incremental, exemplar-free** is the hard standard setting: no stored images, and at test time the task identity is unknown.

1. **New units.** Add $m_t$ units (e.g. $m_t = 64$). Units from tasks $< t$ are **frozen**, so their bits never change and they cause no interference.
2. **New codewords.** For each new class $c \in \mathcal{Y}_t$:
   - *on old units:* $Q_{c,u} = $ the **majority firing** of old unit $u$ on class-$c$ data. The code is defined by what the frozen units already do, so no search is needed;
   - *on new units:* random bits, balanced across $\mathcal{Y}_t$.
3. **Old classes on new units.** Old-class codewords need values on the new units, and the new units must be *trained* to output them without access to old images. Keep per-class feature statistics $(\mu_c, \Sigma_c)$ (or a diagonal $\Sigma_c$) for old classes; this is standard in exemplar-free CIL and costs $O(d)$ or $O(d^2)$ per class, not images. Sample **pseudo-features** $\hat z \sim \mathcal{N}(\mu_c, \Sigma_c)$ and set targets $Q_{c,u} = 0$ for $u$ in the new units.
4. **Search** only the new units' genomes $W_t \in \{-1,0,1\}^{m_t\times d}$, on new-class data plus pseudo-features.

**Memory per task:** $m_t \cdot d \cdot \log_2 3$ bits ≈ $64 \cdot 768 \cdot 1.58 \approx 78$ kbit ≈ 10 kB, plus class statistics.

**Forgetting:** old units are unchanged. What can degrade old-class accuracy is only **decoding confusion** from new codewords. That is measurable, and it is the real continual-learning problem here (task-recency bias in another guise).

### 3.5 Re-adaptation under drift

A stream can also *drift*: the same classes, but features shift (corrupted images, a new camera). The genome $W$ is then re-optimised starting from the current one. The number of positions that need to change, $k = \mathrm{Ham}(W^{\text{old}\star}, W^{\text{new}\star})$, is the natural measure of the size of the shift. §5 asks how search time scales with $k$.

---

## 4. Search and implementation

### 4.1 Algorithms (ordered by how much to trust them first)

1. **Initialisation:** closed form from the linear surrogate (§5.2): $w_u^{(0)} = \mathrm{sign}(v_u)$ with $v_u = \sum_b (2t_{b,u}-1)\, z_b$, keeping only the top-$s$ magnitudes (sparsity $s \in \{16, 64, 256\}$) and zeroing the rest. This is a "which features differ most between the classes that should fire and those that shouldn't" detector, and it is the baseline the GA must improve on.
2. **Per-unit $(1+\lambda)$ EA** with ternary mutation at rate $1/d$ to $4/d$, $\lambda = 256$–$4096$ offspring evaluated as one batched matmul; the threshold is recomputed per offspring by sorting.
3. **Per-unit PBIL / cGA** over $\{-1,0,1\}^d$: population sampled from the categorical probabilities, winners shift the probabilities. The parallel-friendly default.
4. **$(\mu+\lambda)$ GA** with uniform crossover, for the global refinement on true accuracy.

### 4.2 GPU implementation sketch

```python
# Z: [B, d] cached features (float16), T: [B, m_t] target bits
# P: [m_t, d, 3] PBIL probabilities over {-1,0,+1}
for gen in range(G):
    W = sample_ternary(P, n_pop)                 # [m_t, n_pop, d] int8
    S = torch.einsum('bd,upd->bup', Z, W.half())  # [B, m_t, n_pop] projections: one big matmul
    tau, Fu = best_threshold(S, T)               # sort along B per (unit, member): [m_t, n_pop]
    elite = Fu.topk(k, dim=1).indices            # per-unit elites
    P = pbil_update(P, W, elite, lr)             # shift probabilities towards elites
```

**Cost per generation:** $B \cdot d \cdot m_t \cdot N$ multiply-adds. With $B = 5000$ (a subsample), $d = 768$, $m_t = 64$, $N = 1024$, this is ≈ 0.25 TFLOP, i.e. **tens of milliseconds on a 3090**, and it fits in memory with chunking over units. The threshold sort dominates at large $B$; use a histogram approximation with 256 bins. On the MacBook (MPS) it is slower but still workable.

---

## 5. Theory: what can be said

### 5.1 The honest starting point

Accuracy of a single threshold unit, $F_u(w_u)$, is the **0–1 loss of a halfspace**. Finding the optimal halfspace under 0–1 loss is NP-hard in general, even without the ternary restriction. So no EA gets a polynomial-time guarantee for the *true* fitness in the worst case. Theory must therefore target (a) structured surrogates, (b) re-optimisation from a good solution, or (c) measured landscape properties.

### 5.2 The linear surrogate is trivial, and that is informative

The margin surrogate $M_u(w) = \sum_b s_b\langle w, z_b\rangle$ with $s_b = 2t_{b,u}-1 \in \{\pm1\}$ is **linear**:

$$M_u(w) = \langle w, v_u\rangle, \qquad v_u = \sum_b s_b z_b .$$

Under a sparsity budget $\|w\|_0 \le s$, it is maximised in closed form by taking the $s$ largest $|v_{u,j}|$ and setting $w_j = \mathrm{sign}(v_{u,j})$. **No EA is needed for it.** This initialisation is a class-mean-difference detector.

**Consequence:** any value the GA adds must come from the *non-linear* part of the real fitness: thresholding, class imbalance, and interactions between features (epistasis). This is the interesting question, and it is measurable (§5.4).

### 5.3 Re-optimisation bounds (a real theorem, on the idealised case)

For **OneMax-like** fitness (each position contributes independently and equally), the (1+1) EA started at Hamming distance $k$ from the optimum needs expected time

$$\mathbb{E}[T_k] \;\le\; \sum_{i=1}^{k} \frac{e\,n}{i} \;=\; e\,n\,H_k \;=\; O(n \log k)$$

by the **fitness-level method**. At distance $i$ there are $i$ improving single-position changes; each is made alone with probability $\ge \frac{1}{n}(1-\frac1n)^{n-1} \ge \frac{1}{en}$, so an improvement happens with probability $\ge \frac{i}{en}$ per step. Summing the expected waiting times gives $en H_k$.

For a ternary alphabet with mutation to a uniformly random other value, the constant doubles: $\le 2enH_k$. For **general linear functions** with positive weights, tight $O(n\log n)$ bounds from arbitrary initialisation exist (Droste, Jansen & Wegener 2002; Witt 2013). Re-optimisation bounds for linear functions under dynamic constraints exist too (Shi, Schirneck, Friedrich, Kötzing & Neumann, Algorithmica 2019). A *distance-$k$* bound for general linear functions is, to my knowledge, not standard, and a nice self-contained theory question.

**The claim this supports, to be tested empirically:** *adaptation time scales with the size of the change ($\log k$), not with the size of the model ($\log n$).* For continual learning this is exactly the property wanted: small shifts should be cheap. Gradient methods have no analogous clean statement.

### 5.4 Measuring how far reality is from the idealised case

- **Additivity / epistasis test.** Take the evolved optimum $w^\star$ and a set of single-position changes $\delta_j$. Measure $F(w^\star + \delta_i + \delta_j) - F(w^\star + \delta_i) - F(w^\star+\delta_j) + F(w^\star)$ over random pairs. The distribution of these second differences quantifies epistasis. This is the same instrument as the repo's E12 additivity test, now in a setting where it is cheap to measure exactly (no sampling noise, since the fitness is deterministic on cached features).
- **Re-optimisation curves.** For drifts of increasing severity, plot generations-to-recover vs the measured $k$. Fit $a + b\log k$ vs $a + bk$ and compare with the fitness-level prediction.

---

## 6. Prior work explained

- **Supermasks & SupSup (arXiv:1905.01067, 1911.13299, 2006.14769).** A randomly initialised network contains subnetworks, found by learning a binary *mask* over its fixed weights, that perform well without any weight training ("edge-popup" learns a score per weight and keeps the top-$k$ via STE). **Supermasks in Superposition** stores one mask per task: zero forgetting, a few bits per weight per task, and task identity inferred at test time from output entropy. *Relation:* the same "discrete selection over fixed weights, one genome per task" logic, but masks are found by gradient (STE) on *random* weights. F3 uses pretrained features, ternary selection, a GA, and a fixed decoder.
- **Ternary Feature Masks (arXiv:2001.08714).** Per-task ternary masks on feature maps (use / don't use / "new") for task-incremental learning without forgetting. Again gradient-trained.
- **RanPAC (arXiv:2307.02251).** Frozen pretrained ViT → fixed random projection with ReLU to $M \approx 10^4$ dims → a linear classifier obtained in closed form by ridge regression from accumulated Gram statistics $G = \sum \tilde z\tilde z^\top$ and class sums. Since the statistics are simply accumulated, the head is exactly the joint-training solution: no forgetting of the head. It optionally adapts the backbone with PETL on the first task only. **It is the main baseline:** a very strong, closed-form, frozen-feature method. A GA will not beat closed-form least squares on the least-squares objective; it has to win on something else (sparsity and interpretability, non-differentiable objectives, bits per task, drift re-adaptation).
- **SimpleCIL / ADAM (arXiv:2303.07338).** Nearest-class-mean on frozen pretrained features is a surprisingly strong CIL baseline. It is the "zero-learning" floor.
- **L2P (arXiv:2112.08654).** A frozen ViT with a pool of learnable prompts chosen by key–query matching. The prompt pool is a memory, as the archive is here.
- **Weight Agnostic Neural Networks (arXiv:1906.04358)** evolve *topology* with a single shared weight value, showing that structure alone can compute. **Tsetlin machines (arXiv:1804.01508)** are propositional-logic classifiers built from learning automata over boolean features, i.e. discrete, interpretable and non-gradient. **Differentiable logic gate networks (arXiv:2210.08277)** learn networks of binary logic gates by relaxation. All three show discrete computation is viable; none are built on frozen foundation features for continual learning.
- **DFR, "Last Layer Re-Training is Sufficient…" (arXiv:2204.02937).** On features from a standard ERM-trained model, retraining only the *last layer* with logistic regression on a small group-balanced held-out set gives state-of-the-art worst-group accuracy on spurious-correlation benchmarks (Waterbirds, CelebA). This is the secondary benchmark's baseline (§7.2): a strong, cheap last-layer method that optimises a *proxy* (balanced log-loss) of the real target (worst-group accuracy).
- **EA theory:** Droste, Jansen & Wegener (TCS 2002) and Witt (CPC 2013) for linear functions; Kötzing, Lissovoi & Witt (FOGA 2015) on dynamic OneMax variants; Shi, Schirneck, Friedrich, Kötzing & Neumann (Algorithmica 2019) on re-optimisation under dynamic constraints. These are the tools for §5.

---

## 7. Benchmark protocol

### 7.1 Primary: exemplar-free class-incremental learning on frozen ViT features

**Data and features.**

- **Split CIFAR-100**, 10 tasks × 10 classes. **ImageNet-R** (200 classes, renditions), 10 tasks × 20 classes. Both are standard in the L2P / RanPAC / SimpleCIL literature.
- Encoder: ViT-B/16 pretrained on ImageNet-21k (`timm`: `vit_base_patch16_224.augreg_in21k`), CLS token, 768-d. Images are resized to 224 with no augmentation for feature extraction (a single pass).
- Cache train and test features once: ~10 min on a 3090, longer on a MacBook.
- 3 class orders (seeds) at minimum, 5 preferred.

**Arms.**

| # | arm | role |
|---|---|---|
| B0 | NCM / SimpleCIL on the same features | zero-learning floor |
| B1 | RanPAC, frozen (no first-session adaptation), $M = 10^4$ | **strong baseline** |
| B2 | joint linear probe (all classes at once, Adam) | non-continual upper reference |
| B3 | **STE-trained ternary layer**, same architecture, codebook and growth protocol, Adam | **key control:** same model, gradient optimiser |
| B4 | closed-form surrogate init only (§5.2) | how far no search gets |
| B5 | **PBIL per unit + global GA refinement** | the main arm |
| B6 | B5 with random expansion $\tilde z = \mathrm{ReLU}(Rz)$, $M = 2048$ | expansion helps? |
| B7 | B5 with fitness = worst-class accuracy (global phase) | non-differentiable objective |
| B8 | random search at matched evaluations, own seeds | noise control |

**Hyperparameters (starting points):** $m_t = 64$ units per task (sweep 32–256); sparsity $s = 64$; PBIL population 1024, learning rate 0.1, 50–200 generations per unit batch; global refinement: $(\mu+\lambda)$ GA with $\mu = 16$, $\lambda = 256$, 50 generations.

**Metrics.**

- **Final average accuracy** $A_T$ (accuracy on all classes seen so far, after the last task) and **average incremental accuracy** $\bar A = \frac1T\sum_t A_t$;
- **forgetting** per class-group: the drop from its best accuracy to its final accuracy;
- **bits per task** and total model size after the encoder;
- **wall-clock** per task;
- **interpretability probe:** mean number of non-zero weights per unit, and whether units align with human-readable concepts (optional: project the features onto CLIP text directions).

**Drift sub-experiment (the theory test).** After task 10, present CIFAR-100-C corruptions (gaussian noise, blur, contrast; severities 1–5) of the test-split classes as a *drift*, with a small labelled adaptation set (10 per class). Re-optimise all units starting from the current genome. Measure:

- generations to recover 95% of the pre-drift accuracy, per severity;
- the Hamming distance $k$ between old and new optima;
- a fit of recovery time vs $\log k$;
- the same for STE (steps to recover) as the comparison.

**Compute:** feature extraction is minutes to an hour. Each full CIL run takes minutes to tens of minutes on a 3090 or 1080 Ti. **The entire study fits in a few GPU-days, and most of it runs on a laptop.**

### 7.2 Secondary: direct optimisation of worst-group accuracy (Waterbirds)

- **Why:** a clean case where the target metric (worst-group accuracy) is non-differentiable and the best-known method (DFR) optimises a *proxy*. If the GA wins anywhere on accuracy, it should be here.
- **Setup:**
  - Waterbirds (landbird/waterbird × land/water background; four groups, where the minority groups are the spurious-correlation-breaking ones);
  - features from an ERM-trained ResNet-50, following DFR's recipe (use their released code; training ERM yourself takes ~1 GPU-hour);
  - split the *validation* set in half: one half is the GA's fitness set (group labels available, as in DFR), the other half is for model selection;
  - report on the test set.
- **Arms:**
  - DFR (logistic regression on the group-balanced half);
  - ternary layer + codebook, trained by GA directly on worst-group accuracy;
  - the same with STE on balanced cross-entropy;
  - a GA over a *float* linear head (CMA-ES, $n = 2 \times 2048$) on worst-group accuracy. This isolates "discrete" from "direct objective".
- **Metrics:** worst-group accuracy, mean accuracy, and the gap between them.

---

## 8. Hypotheses, risks, kill criteria

**Hypotheses.**

- **H1:** the GA ternary layer (B5) matches the STE ternary layer (B3) within 1–2 points of final accuracy, at comparable wall-clock.
- **H2:** B5 is within ~5 points of RanPAC (B1) while using **≥50× fewer bits** after the encoder.
- **H3:** under drift, recovery generations grow like $\log k$, not linearly in $k$, and faster than STE re-training at matched compute.
- **H4 (secondary):** direct GA optimisation of worst-group accuracy ≥ DFR on Waterbirds. The float CMA-ES arm tells whether this is due to the discrete layer or to the direct objective.
- **H5:** measured epistasis is low (second differences concentrated near 0). This is what makes a GA effective and the theory relevant.

**Risks.**

- **Codeword design for class-incremental:** old/new codeword confusion may dominate the error. Diagnose with a confusion matrix split into old→new and new→old errors, and consider a per-task bias correction on the decoding distance.
- **Pseudo-feature quality:** Gaussian class statistics may be poor on ViT features. The fallback is the task-incremental setting (task id known), where this problem disappears.
- **RanPAC is very strong:** closed-form ridge regression on $10^4$ random features. Losing to it on raw accuracy is likely and fine *if* H2's efficiency and H3's adaptation results hold.

**Kill criteria.**

- If **H1 fails** by more than ~5 points at matched wall-clock, a discrete GA is not competitive with STE for this layer. Keep only the theory contribution (§5 and the drift re-optimisation study), which is still a FOGA/GECCO-theory-track shaped paper.
- If **H5 fails** (strong epistasis everywhere), the OneMax-style theory is irrelevant for this model class. Pivot the theory part to measuring landscape structure.

---

## 9. Extensions

- **A learned decoder instead of ECOC:** map $h$ into CLIP's text-embedding space via a fixed random ±1 matrix and decode by nearest class-name embedding. This gives zero-shot new classes: a new class needs only a name.
- **Logic layer on top of units:** a second evolved layer of AND/OR gates over the unit outputs (cf. Tsetlin machines, logic gate nets) for compositional concepts.
- **Plastic thresholds** (link to F5): let $\tau_u$ adapt online by a local rule, e.g. homeostasis targeting a firing rate, so that drift is handled without search.
- **LLM version:** frozen LLM hidden states as $z$, ternary units as probes, and a codebook over answer options. This is a GA-trained discrete probe for classification tasks.
