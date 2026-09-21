# F5 — Evolved Plasticity: Evolve the Learning Rule of the Late Layer, Not Its Weights

**Part of:** F — The Late Evolutionary Layer (`research/base/base.tex` has the shared notation and the general theory)
**Status:** Proposal (stretch) · no results yet · **Date:** 2026-09-21
**One-line pitch:** *Instead of searching the late layer's weights for every new context, let evolution search once, over a distribution of contexts, for a **local learning rule**. At deployment, the late layer then adapts itself continuously, with no population, no archive and no outer search: one forward pass plus one local update per step.*

F1–F4 all run the EA **at deployment time**: every new context triggers a search. F5 moves the EA **one level up**: it runs offline, once, and produces a *learner*. It is the most "self-adapting" of the five ideas and the most speculative.

---

## 0. Reading guide

- §1 background: Hebbian plasticity, neuromodulation, fast weights, and meta-learning by ES;
- §2 the idea;
- §3 the math: rule family, lifetime, outer objective, and why the family must contain SGD;
- §4 the algorithm;
- §5 prior work;
- §6 the benchmark;
- §7 hypotheses, risks and kill criteria;
- §8 extensions.

---

## 1. Background

### 1.1 Hebbian plasticity

"Neurons that fire together wire together." A synapse from presynaptic neuron $j$ (activity $o_j$) to postsynaptic neuron $i$ (activity $o_i$) changes as a function of local quantities only:

$$\Delta w_{ij} = \eta\, o_i\, o_j .$$

Plain Hebb is unstable (weights only grow). The **generalised "ABCD" rule** adds correlation-free terms:

$$\Delta w_{ij} = \eta_{ij}\,\big(A_{ij}\, o_i o_j + B_{ij}\, o_i + C_{ij}\, o_j + D_{ij}\big).$$

$A, B, C, D, \eta$ are the **rule parameters**. Different values give Hebbian, anti-Hebbian, pre-/post-synaptic decay-like behaviour, and so on.

### 1.2 Neuromodulation: making plasticity task-directed

Pure Hebbian rules learn correlations but don't know what is *good*. Brains gate plasticity with **neuromodulators** (e.g. dopamine, which signals reward-prediction error). In models, a scalar or vector **modulatory signal** $M(t)$ multiplies the update:

$$\Delta w_{ij} = M(t)\cdot \eta_{ij}\,(A_{ij} o_i o_j + B_{ij} o_i + C_{ij} o_j + D_{ij}).$$

$M(t)$ can be the reward minus a running baseline (RL), the prediction error (supervised) or a learned signal.

### 1.3 The delta rule is a special case (important)

For a linear output $o = Wz$, target $y$ and squared error, SGD gives

$$\Delta W = \eta\,(y - o)\,z^\top \quad\Longleftrightarrow\quad \Delta w_{ij} = \eta\,(y_i - o_i)\, z_j = \eta\,(y_i z_j - o_i z_j).$$

This is a modulated-Hebbian rule: an $A$-term with a negative sign, plus a "target × input" term, if the target (or the error $e_i = y_i - o_i$) is available as a local signal. **So a rule family that includes error-modulated terms contains SGD on the last layer.** This matters for honesty: an evolved rule can only beat a tuned SGD head if it finds something SGD doesn't do, such as per-feature learning rates, decay or forgetting schedules, sign asymmetries, or homeostasis.

### 1.4 Meta-learning by evolution strategies

- **Inner loop ("lifetime"):** the network starts from initial weights $W_0$ and runs on a stream; its weights change by the rule.
- **Outer loop ("evolution"):** ES perturbs the rule parameters $\theta$ and scores each by the *total performance over the lifetime*, averaged over many sampled contexts.
- Since the inner loop can be long and non-differentiable, ES is the natural outer optimiser. It needs only lifetime returns, with no backprop through time, and it is embarrassingly parallel over (member, context) pairs.

---

## 2. The idea

```
                   frozen backbone φ  (from F1–F3: ViT features, LLM hidden states, proprioceptive encoder)
 x_t ──► z_t = φ(x_t) ──► late layer  o_t = W_t z_t  ──► ŷ_t / a_t
                               ▲
                               │  W_{t+1} = W_t + ΔW_t,   ΔW_t = Rule_θ(o_t, z_t, M_t)   ← local, O(md) per step
                               │
                        modulatory signal M_t (error / reward − baseline)

 OUTER (offline, evolutionary):   θ* = argmax_θ  E_{context c}  [ Σ_t performance_t  over a lifetime in c ]
                                  ES over θ, population evaluated on many contexts in parallel
```

- **Genome:** rule parameters $\theta$ (and optionally $W_0$), shared or factorised so that $n$ stays small.
- **Deployment:** there is no search. The late layer keeps adapting as the stream goes, which gives continual learning *by design*, driven by what evolution found works across a distribution of drifts.
- **Relation to F1–F4:** F5 replaces "search + archive" with "learned learning". A hybrid is also possible: the archive stores rule parameters per *kind* of context.

---

## 3. Math

### 3.1 Rule family

With late-layer weights $W \in \mathbb{R}^{m\times d}$, pre-activations $z_t \in \mathbb{R}^d$ (frozen features), post-activations $o_t = \sigma(W_t z_t) \in \mathbb{R}^m$ and a modulatory signal $M_t \in \mathbb{R}^m$ (per output unit, e.g. an error $e_t = y_t - o_t$ in the supervised case):

$$\Delta W_t = \eta \odot \Big[\; M_t \big(\underbrace{A \odot (o_t z_t^\top)}_{\text{Hebbian}} + \underbrace{B \odot (o_t \mathbf{1}^\top)}_{\text{post}} + \underbrace{C \odot (\mathbf{1} z_t^\top)}_{\text{pre}} + D\big) \;+\; \underbrace{E \odot (e_t z_t^\top)}_{\text{delta-rule term}} \;\Big] \;-\; \underbrace{\lambda \odot W_t}_{\text{decay}}$$

Here $\odot$ is elementwise and the modulated terms broadcast $M_t$ along rows. The delta-rule term guarantees that the family contains SGD-on-the-last-layer (§1.3).

**Parameter sharing (keeping $n$ small).** Full per-synapse parameters give $6md$ values, which is too many for ES and invites overfitting. Use **factorised** parameters: $A_{ij} = a_i\, a'_j$, and likewise for the rest. Per-output factors $a \in \mathbb{R}^m$ and per-feature factors $a' \in \mathbb{R}^d$ give $n = 6(m + d) + (m+d)$ for $\eta$ and $\lambda$. With $m = 100$ and $d = 768$ that is $n \approx 6{,}000$: fine for sep-CMA-ES or EGGROLL-style ES. A fully *shared* version (scalars $A..E, \eta, \lambda$, so $n = 7$) is the minimal baseline.

### 3.2 Lifetime and outer objective

A **context** $c \sim p(\mathcal{C})$ defines a non-stationary stream $\{(x_t, y_t)\}_{t=1}^{T_{\text{life}}}$, e.g. a sequence of label remappings or feature corruptions with switch points. The lifetime performance is

$$\mathcal{J}_c(\theta) = \frac{1}{T_{\text{life}}}\sum_{t=1}^{T_{\text{life}}} \mathrm{perf}\big(\hat y_t, y_t\big), \qquad \text{with } \hat y_t \text{ predicted \emph{before} the update at step } t \text{ (online / prequential evaluation).}$$

The outer objective is

$$\max_\theta\; \mathcal{J}(\theta) = \mathbb{E}_{c\sim p(\mathcal{C})}\big[\mathcal{J}_c(\theta)\big].$$

**Prequential** (predict first, then learn) scoring rewards *fast adaptation after each switch*, not just final accuracy. This is exactly the property wanted.

### 3.3 Outer ES

$$\theta \leftarrow \theta + \frac{\alpha}{N\sigma}\sum_{i=1}^{N} u_i\, \epsilon_i, \qquad \epsilon_i\sim\mathcal{N}(0,I),$$

where $u_i$ are rank-based utilities of $\hat{\mathcal{J}}(\theta + \sigma\epsilon_i)$, estimated on a mini-batch of contexts that is **shared across members** (common random numbers: the same contexts and the same streams). Antithetic pairs $\pm\epsilon_i$ halve the variance. Alternatively, use CMA-ES for the $n = 7$ shared rule or sep-CMA for the factorised one.

**Parallel structure.** $N$ members × $K_c$ contexts × $T_{\text{life}}$ steps. Each step is one small matmul plus an outer-product update, so a lifetime is a `scan`, and all (member, context) pairs are batched on one GPU in JAX (`vmap` over members and contexts).

### 3.4 Generalisation split

Contexts are split into meta-train and meta-test *families*. For example: meta-train on label permutations and blur/noise corruptions; meta-test on unseen corruption types and **class-incremental** additions. The claim is only interesting if $\theta^\star$ transfers to unseen kinds of drift.

---

## 4. Algorithm

```
Outer (offline):
  θ ← init (e.g. pure delta rule: E=1, others 0, η tuned)      # start from SGD!
  for gen = 1..G:
      contexts C_batch ← sample K_c contexts (shared by all members)
      for i = 1..N (parallel):  J_i ← mean_{c ∈ C_batch} Lifetime(θ + σ ε_i, c)
      θ ← ES/CMA update with ranks of J_i

Lifetime(θ, c):
  W ← W_0 ;  score ← 0
  for t = 1..T_life:
      z ← φ(x_t)  (cached) ;  o ← σ(W z) ;  score += perf(o, y_t)     # predict first
      e ← y_t − o ;  M ← modulator(e or reward)
      W ← W + Rule_θ(o, z, M, e) − λ⊙W                                  # then learn
  return score / T_life
```

**Initialising at the delta rule** means the outer search starts *at* tuned SGD. Any improvement is then something beyond SGD, which makes the key control (a tuned SGD head) automatically part of the search trajectory.

---

## 5. Prior work explained

- **Najarro & Risi, "Meta-Learning through Hebbian Plasticity in Random Networks" (arXiv:2007.02686, NeurIPS 2020).**
  - They evolve per-synapse ABCD rule parameters with ES; network weights are **randomly initialised at the start of every episode** and self-organise within the episode by the rule.
  - Tasks: CarRacing from pixels and a simulated quadruped.
  - The Hebbian networks adapted within an episode and coped with morphological damage **not seen during evolution**, where static-weight networks failed.
  - *Relation:* the direct ancestor. F5 differs by (i) placing plasticity only in a late layer on top of a strong frozen backbone, (ii) including an error-modulated term so SGD is in the family, and (iii) evaluating on continual-learning streams.
- **Differentiable plasticity / Backpropamine (Miconi et al., arXiv:1804.02464, arXiv:2002.10585):** Hebbian traces with learned plasticity coefficients and neuromodulation, meta-trained by **backprop** through the lifetime. It is the gradient-based counterpart: fine for short lifetimes, but expensive or unstable for long ones, where ES shines.
- **Fast weights (Ba et al., arXiv:1610.06258):** a rapidly changing weight matrix updated by outer products of recent activations, i.e. attention-like short-term memory. The same mathematical object as a Hebbian late layer.
- **CAVIA (arXiv:1810.03642) and ES-MAML (arXiv:1910.01215):** meta-learning that adapts only a small context vector (CAVIA), and MAML with ES instead of second-order gradients (ES-MAML). The outer-loop ES machinery used here.
- **Online continual learning with frozen pretrained features:** streaming NCM, streaming LDA and similar methods are strong, simple baselines when features are good. They are the practical bar for §6.

---

## 6. Benchmark protocol

### 6.1 Stage A (cheap, start immediately): drifting streams on cached ViT features

Reuse F3's cached ViT-B/16 features for CIFAR-100 (and ImageNet-R). Define **context families**, each a stream of $T_{\text{life}} = 2000$ labelled samples in which something changes every 200–500 steps:

| family | what drifts | split |
|---|---|---|
| F-perm | label permutation among 10 classes | meta-train |
| F-corrupt | feature distribution: features of corrupted images (CIFAR-100-C: noise, blur) | meta-train: 2 types; meta-test: 3 other types |
| F-new | new classes appear (class-incremental within the stream) | meta-test only |
| F-mix | random combination of the above | meta-test |

**Arms** (all prequential, all starting from the same $W_0$):

| # | arm | role |
|---|---|---|
| P0 | static head trained on the first segment | no adaptation |
| P1 | **online SGD (delta rule), learning rate and decay tuned per family** | **key control** |
| P2 | streaming NCM with exponential forgetting (tuned) | strong simple baseline |
| P3 | evolved shared rule ($n = 7$) | minimal plasticity |
| P4 | evolved factorised rule ($n\approx 6$k) | main arm |
| P5 | P4 without the $E$ (delta) term | does Hebbian-only work? |
| P6 | F3-style re-search after each switch (GA on the layer, given the switch time) | search-at-deployment reference |

**Metrics:**

- prequential accuracy over the lifetime;
- **recovery time** after a switch (steps to reach 90% of the pre-switch accuracy);
- accuracy on classes from before the last switch (forgetting);
- the meta-train → meta-test gap.

**Compute:** lifetimes are tiny (2000 steps × a 768×100 matmul). In JAX, $N = 256$ members × 32 contexts per generation takes seconds, and 1000 generations take under an hour on one GPU. This runs on the 1080 Tis.

### 6.2 Stage B (optional): damaged locomotion

F4's Brax Ant with a frozen proprioceptive encoder (or none). The plastic layer is the policy's last layer, the modulator is reward minus a running baseline, and lifetimes are episodes with a damage injected mid-episode. Compare with Najarro & Risi-style fully Hebbian networks and with F4's search-based adaptation, on *unseen* damages.

---

## 7. Hypotheses, risks, kill criteria

**Hypotheses.**

- **H1:** P4 > P1 (tuned SGD) on meta-test families in prequential accuracy, primarily through faster recovery after switches.
- **H2:** the gain transfers from meta-train to meta-test families (unseen corruption types and class-incremental).
- **H3:** P5 (no delta term) is clearly worse than P4. Error modulation is necessary, and pure Hebbian learning is insufficient on top of frozen features.

**Risks.**

- **Tuned SGD is hard to beat on a linear head,** since it is close to optimal for stationary segments. The evolved rule must win on *transitions* (forgetting, re-weighting), which is where per-feature rates and decay can help.
- **Meta-overfitting to the drift families:** the meta-test split is essential.
- **Variance of lifetime returns:** use common random numbers (shared contexts per generation) and antithetic sampling.

**Kill criteria.** If P4 does not beat P1 by ≥2 sem in prequential accuracy on meta-test families, the evolved rule is not finding anything beyond SGD in this setting. Keep only Stage B, where the reward-modulated setting differs from SGD, if robotics is being pursued.

---

## 8. Extensions

- **Plastic thresholds for F3's discrete units:** a homeostatic rule on $\tau_u$ (target firing rate), evolved.
- **Evolved rules + archive:** store rule parameters per *family* of contexts and recall by key, i.e. a hierarchy of fast weights, rules and an archive of rules.
- **Plastic mixing coefficients for F2:** the composition genome $\alpha$ updated online by an evolved rule from the per-example error, i.e. learned test-time adaptation of the composition.
