# F4 — Repertoire Adaptation: Evolving a Tiny Genome over a Quality-Diversity Skill Library, Inside a Learned Model

**Part of:** F — The Late Evolutionary Layer (`research/base/base.tex` has the shared notation and the general theory)
**Status:** Proposal · no results yet · **Date:** 2026-09-21
**One-line pitch:** *Offline, evolve a large and diverse library of walking behaviours for a simulated robot. When the robot gets damaged, search a ~20-number genome that selects and blends library behaviours. Run that search massively in parallel inside a learned model of the damaged robot, and spend only a handful of real trials. Remember each damage's solution.*

This is the **robotics / planning** bullet of the whiteboard. It is the most engineering-heavy idea, and the one that must confront the fact that **real robots cannot run 10,000 trials in parallel**.

---

## 0. Reading guide

- §1 background: quality-diversity (MAP-Elites), Intelligent Trial and Error, and model-based policy search;
- §2 the idea;
- §3 the math;
- §4 the algorithm;
- §5 prior work;
- §6 the benchmark;
- §7 hypotheses, risks and kill criteria;
- §8 extensions.

---

## 1. Background

### 1.1 Quality-Diversity (QD) and MAP-Elites

A normal optimiser returns **one** best solution. A QD algorithm returns a **map of many** high-performing solutions that each *behave differently*.

**MAP-Elites** (Mouret & Clune 2015):

- Define a **behaviour descriptor** $d(\pi) \in \mathcal{B} \subset \mathbb{R}^D$ that says *how* a solution behaves. For a legged robot, a standard choice is the **duty factor** of each foot: the fraction of time it touches the ground.
- Discretise $\mathcal{B}$ into cells, e.g. a grid or a CVT (centroidal Voronoi tessellation) with ~1000–10,000 cells.
- Loop: pick random elites from the map, mutate them, evaluate fitness $f$ and descriptor $d$, and place each offspring in its cell if the cell is empty or the offspring beats the incumbent.
- **Output:** a **repertoire** $\mathcal{P} = \{(\pi_j, d_j, f_j)\}$, i.e. for every way of walking, the best-known controller that walks that way.

**Why this matters for adaptation.** When the robot is damaged, the best *nominal* gait might rely on the broken leg, but some *other* gait in the map, e.g. one that barely uses that leg, probably still works. The repertoire is a library of diverse "experts". This is the robotics analogue of F2's diversified LoRA library.

### 1.2 Intelligent Trial and Error (ITE) — Cully et al., Nature 2015 (arXiv:1407.3501)

- **Offline:**
  - a hexapod robot and a periodic open-loop controller with 36 parameters;
  - the descriptor is the 6-D duty factor (one per leg);
  - MAP-Elites in simulation produces a map of ~13,000 gaits, which took a large amount of offline simulation.
- **Online, after damage** (a broken or missing leg, on the real robot), **Bayesian optimisation over the map:**
  - a Gaussian process $\hat f(d)$ models the *real* performance of the gait in cell $d$;
  - its **prior mean is the simulated performance** $f_j$ from the map;
  - its kernel (Matérn) says that nearby behaviours perform similarly;
  - repeat: pick the cell maximising $\mu(d) + \kappa\sigma(d)$ (UCB), run it on the real robot, update the GP;
  - stop when the best observed performance is ≥ 90% of the best predicted.
- **Result:** the robot recovered effective gaits within about two minutes and a small number of real trials, across many damage conditions.
- **Why it works:** the search space is the **low-dimensional descriptor space** ($D = 6$), not the 36-D (or larger) controller space. Diversity was paid for offline, and adaptation is a tiny search online. This is the whiteboard's thesis in its cleanest proven form.

### 1.3 Model-based policy search with CMA-ES (Black-DROPS, arXiv:1703.07261)

- Learn a **dynamics model** $\hat s_{t+1} = \hat f(s_t, a_t)$ from the real transitions seen so far (Black-DROPS uses Gaussian processes).
- Optimise the policy parameters by **CMA-ES on returns predicted by rolling out the model**, never on the real system. Model rollouts are cheap and parallel.
- Execute the best policy on the real system, add the data, refit the model, repeat.
- It treats the model's predicted return as a *noisy black-box* fitness, and CMA-ES handles the noise. It was data-efficient (few real episodes) and parallelised over CPU cores.

### 1.4 Why GPU simulators change the picture

**Brax / MJX** (JAX physics) run thousands of environments in parallel on one GPU, and **QDax** implements MAP-Elites and friends on top of them. The offline repertoire, which took ITE a long time in 2015, now takes **hours on one GPU**. Model rollouts in JAX are equally parallel. So the EA's natural strength, many cheap parallel evaluations, is available wherever evaluation is *simulated or modelled*.

---

## 2. The idea

```
 OFFLINE (slow, diverse)                       ONLINE after damage / change (fast, tiny genome)
 ─────────────────────                         ────────────────────────────────────────────────
 MAP-Elites in Brax (QDax)                      real trials (scarce)        learned model (cheap, parallel)
   ► repertoire 𝒫 = {(π_j, d_j, f_j)}            ┌───────────────┐          ┌───────────────────────────┐
     ~1000–5000 diverse gaits                    │ execute best g │─ data ─►│ residual dynamics δ_ω(s,a) │
                                                 └───────▲───────┘          │ (ensemble of small MLPs)   │
                                                         │                  └────────────┬──────────────┘
                                                         │                               ▼
                                                         │     CMA-ES over genome g (n≈10–50):
                                                         └──── thousands of model rollouts per generation
                                                                                         │
                                         archive 𝓜 = {(damage key κ, g*)} ◄──────────────┘
```

- **Slow system:** the repertoire (diverse expert gaits) and the nominal simulator.
- **Fast system:** a genome $g$ that picks a region of the repertoire, blends a few gaits and adds a small feedback correction.
- **Where the population runs:** in a **learned residual model** of the damaged robot, so the EA is massively parallel where evaluation is cheap and frugal where it is expensive.
- **Memory:** an archive of solved damages. A new damage first tries the archived genomes, which is one parallel evaluation in the model or a few real trials.

"Real" here means a **held-out damaged simulator** with a trial budget: the standard stand-in in this literature. A real robot is an optional later stage.

---

## 3. Math

### 3.1 Repertoire

$\mathcal{P} = \{(\pi_j, d_j, f_j)\}_{j=1}^{J}$, where:

- $\pi_j$ is a controller (a closed-loop MLP policy $\pi_j(a\mid s)$ in QDax's Brax tasks, or an open-loop periodic controller as in ITE);
- $d_j \in [0,1]^D$ is the descriptor (foot duty factors: $D = 4$ for Ant, $D = 6$ for a hexapod);
- $f_j$ is the nominal fitness (forward distance).

### 3.2 Genome

$$g = \big(d^\star \in [0,1]^D,\; \log T \in \mathbb{R},\; \kappa \in \mathbb{R}^{n_{\text{fb}}}\big), \qquad n = D + 1 + n_{\text{fb}} \approx 10\text{–}50 .$$

- **Blend.** Let $\mathcal{N}_q(d^\star)$ be the $q$ repertoire entries nearest to $d^\star$ (e.g. $q = 4$), with softmax weights $w_j(g) \propto \exp\!\big(-\|d_j - d^\star\|^2 / T\big)$.
- **Policy:**

$$\pi_g(s) = \sum_{j\in\mathcal{N}_q(d^\star)} w_j(g)\, \pi_j(s) \;+\; K_\kappa\, \phi_{\text{fb}}(s).$$

  The first term is an action-space blend of neighbouring gaits. The second is a small linear feedback correction on a few proprioceptive features $\phi_{\text{fb}}(s)$ (e.g. body roll/pitch, per-leg joint errors), with gains $K_\kappa$ parameterised by $\kappa$ (e.g. a diagonal or low-rank matrix).
- **Special case $T\to 0$, $\kappa = 0$:** pick the single nearest gait, which is ITE's search space. The genome **strictly generalises ITE**: blending and feedback let it reach behaviours *between* and *around* repertoire cells.

**Why blend in action space?** Neighbouring cells in a MAP-Elites map are often *unrelated* parameter vectors, so parameter-space averaging would produce garbage. Action-space averaging of neighbouring *behaviours* is smoother. It is still not guaranteed to be stable, which is exactly what the model-based evaluation is for.

### 3.3 Learned residual model

Real (damaged) transitions $\mathcal{D}_{\text{real}} = \{(s_t, a_t, s_{t+1})\}$ are few. Instead of learning dynamics from scratch, learn the **residual** relative to the nominal simulator $\mathrm{Sim}$:

$$\hat s_{t+1} = \mathrm{Sim}(s_t, a_t) + \delta_\omega(s_t, a_t), \qquad \delta_\omega \in \{\delta_{\omega_1},\dots,\delta_{\omega_E}\}\ \text{(ensemble of } E = 5 \text{ small MLPs)}.$$

Train each member by MSE on $\mathcal{D}_{\text{real}}$ (with bootstrap resampling) and use the ensemble spread as uncertainty. Before any real data, $\delta = 0$ and the model is the nominal simulator.

**Pessimistic fitness in the model:**

$$\hat F(g) = \frac1E\sum_{e=1}^E R_e(g) \;-\; \beta\,\mathrm{std}_e\big(R_e(g)\big),$$

where $R_e(g)$ is the return of $\pi_g$ rolled out in the model with ensemble member $e$, from the same initial states (common random numbers). The penalty $\beta$ avoids exploiting model errors, the standard failure of model-based search.

### 3.4 The adaptation loop

For real trial $t = 1, 2, \dots, B_{\text{real}}$:

1. Fit $\delta_\omega$ on $\mathcal{D}_{\text{real}}$.
2. Run CMA-ES on $\hat F$ for $G$ generations × $N$ members. Each member is $E$ rollouts, all in parallel on the GPU.
3. Execute $g_t = \arg\max \hat F$ once on the real (held-out damaged) system; record the return $R^{\text{real}}_t$ and the transitions.
4. Stop when $R^{\text{real}}_t$ reaches the target, or when the budget is exhausted.

**Archive / continual:** $\mathcal{M} = \{(\kappa_j, g^\star_j)\}$, with key $\kappa_j$ = the flattened parameters of a *small linear* residual model fitted on the first real episode, or the vector of real returns of a fixed probe set of 4 genomes. For a new damage: evaluate archived genomes **in the model after one real episode**, start CMA-ES from the best, and continue.

### 3.5 Where the parallelism goes (cost accounting)

- **Real trials:** sequential and scarce. This is the metric that matters.
- **Model evaluations:** $N \times E \times G$ rollouts per real trial, e.g. $64 \times 5 \times 50 = 16{,}000$ rollouts of 1000 steps. In Brax/JAX this takes seconds to minutes on one GPU.
- **Memory:** the repertoire policies are small MLPs ($J = 1024$ policies × ~10k params = 10M floats), resident on the GPU and gathered per member.

This is the "exploit swarm computing" point made concrete: **the population lives in a model; the real world only sees the champion.**

---

## 4. Algorithm

```
Offline:
  𝒫 ← MAP-Elites(Brax Ant, descriptor = foot duty factors, CVT 1024 cells, ~1e7 evals)

Online, for each new damage condition:
  D_real ← ∅ ;  δ ← 0
  if archive 𝓜 non-empty:  run 1 real episode with the nominal-best gait (collect data), fit δ,
                            G_rec ← 𝓜 genomes ;  m ← argmax_{g ∈ G_rec} F̂(g)
  else:                     m ← (descriptor of nominal best, log T = −2, κ = 0)
  for t = 1..B_real:
       fit ensemble δ_ω on D_real
       CMA-ES(m, σ) for G generations on F̂ (batched model rollouts)
       g_t ← best ; execute on real system ; D_real ← D_real ∪ transitions
       m ← g_t
       if R_real(g_t) ≥ target: break
  𝓜 ← 𝓜 ∪ {(κ(D_real), g_t)}
```

---

## 5. Prior work explained

- **ITE (arXiv:1407.3501):** §1.2. The baseline, and the conceptual ancestor.
- **Reset-free Trial-and-Error (RTE, arXiv:1610.04213):** extends ITE to recovering *while continuing to operate* (no manual resets between trials). A repertoire of low-level primitives with a GP correction on their outcomes is used by a planner (MCTS) to reach goals despite damage.
- **APROL (arXiv:1907.07029):** keeps *several* repertoires, each generated for a different condition (e.g. terrain), and chooses online which one to use as the prior. This is already a primitive **archive of repertoires**, the closest existing thing to this proposal's memory.
- **Black-DROPS (arXiv:1703.07261):** §1.3. The source of "CMA-ES inside a learned model".
- **QDax (arXiv:2202.01258, arXiv:2308.03665):** a JAX library for MAP-Elites and variants (PGA-ME, DCG-ME, …) with Brax tasks (e.g. `ant_uni`, `ant_omni`, `walker2d_uni`) and GPU-parallel evaluation. Infrastructure for the offline phase and for the parallel model rollouts.
- **Domain randomisation:** train one policy across randomly sampled damages or dynamics. A strong baseline *when the damage distribution is known in advance*; it cannot handle damages outside that distribution.
- **Najarro & Risi, Hebbian policies (arXiv:2007.02686):** robust to morphological damage through within-episode plasticity. See F5; it is an alternative route to the same goal.

---

## 6. Benchmark protocol

### 6.1 Environment and damages

- **Robot:** Brax Ant (4 legs, 8 actuators) via QDax's `ant_uni` task (forward distance fitness; descriptor = 4 foot-contact duty factors). Episode length 1000 steps.
- **Damage set (held out from repertoire construction):**
  - (D1) one leg disabled (actuator gains → 0): 4 variants;
  - (D2) two adjacent legs disabled: 4 variants;
  - (D3) all actuators on one leg at 30% strength: 4 variants;
  - (D4) one knee joint locked (range → 0): 4 variants;
  - (D5) +50% torso mass;
  - 17 conditions in total.
- **"Real" system:** the damaged simulator with **observation noise** and a **different random seed** than the model, and a budget of $B_{\text{real}} = 20$ episodes.
- **Oracle:** MAP-Elites re-run *on the damaged robot* with a full budget. This is the best achievable within the controller class and is used to normalise scores.

### 6.2 Repertoire

QDax MAP-Elites (or PGA-ME for better quality) with a 1024-cell CVT and policy MLPs (2 hidden layers × 64), 5–10M evaluations. That is a few hours on a single 3090/4090. Build 3 repertoires with different seeds.

### 6.3 Arms

| # | arm | real episodes used |
|---|---|---|
| R0 | nominal best gait, no adaptation | 1 |
| R1 | **ITE**: GP-UCB over repertoire cells, prior mean $f_j$, Matérn-5/2 kernel on $d$, stop rule as in Cully et al. | ≤20 |
| R2 | CMA-ES on the genome **directly on real episodes** (no model; sequential, $N = 8$) | ≤20 (2–3 generations) |
| R3 | **ours**: CMA-ES on the genome inside the learned residual model | ≤20 |
| R4 | R3 with $T\to0$, $\kappa = 0$ (selection-only genome) | ≤20 (isolates blend + feedback) |
| R5 | PPO fine-tuning from the nominal best policy | ≤20, and a curve up to 500 |
| R6 | domain-randomised PPO policy (trained on D1–D3-like damages, tested on all incl. D4/D5) | 1 |
| R7 | **ours + archive** on a *sequence*: all 17 damages in random order, then 8 repeats | ≤20 each |

### 6.4 Metrics

- **Normalised performance after $k$ real episodes**, $\text{perf}_k = R^{\text{real}}_k / R_{\text{oracle}}$, for $k = 1, 2, 5, 10, 20$ (median and IQR over damages × seeds).
- **Episodes to reach 80% of the oracle**, capped at 20.
- **Model-exploitation gap:** the predicted return $\hat F(g_t)$ vs the real return, i.e. how much the model over-promises.
- **Archive effect (R7):** episodes-to-80% on repeated damages vs first encounters.
- **Compute:** GPU-minutes per real episode.

### 6.5 Compute and plan

- Repertoire: a few GPU-hours × 3 seeds.
- Adaptation runs: 17 damages × 8 arms × 5 seeds ≈ 700 runs of ≤20 real episodes plus model search. Roughly 2–4 GPU-days in JAX.
- Environment setup (JAX, QDax, damage injection, residual-model training) is realistically **3–4 weeks** before the first real comparison. Treat this idea as a *second-half* project, unless robotics is the goal.

---

## 7. Hypotheses, risks, kill criteria

**Hypotheses.**

- **H1:** R3 reaches 80% of the oracle in fewer real episodes than R1 (ITE) on the damages *outside* the repertoire's comfort zone (D3–D5), where blending and feedback matter.
- **H2:** R3 > R4. The genome's blend and feedback terms add value beyond ITE-style selection.
- **H3:** R7's repeated damages recover in ≤3 real episodes on average.

**Risks.**

- **ITE is very strong** in low dimensions with a good prior. On D1/D2 (missing legs) it may be as good as anything.
- **Model exploitation:** CMA-ES is excellent at finding model errors. The ensemble penalty $\beta$ is critical, so sweep it.
- **Action-space blending can be unstable:** mitigate with $q = 1$–$2$ and a small $T$ initially.
- **Engineering time** (see §6.5).

**Kill criteria.** If R3 does not beat R1 by ≥2 real episodes (median, episodes-to-80%) on D3–D5 after reasonable tuning, the learned-model EA machinery isn't earning its complexity. Report ITE + archive (R1 with memory) as the simpler recommendation.

---

## 8. Extensions

- **Descriptors learned instead of hand-designed** (AURORA-style autoencoders on trajectories), so the genome space is learned too.
- **Hierarchical genome:** a high-level genome chooses a *sequence* of repertoire skills (navigation), as in RTE, but searched by CMA-ES in the model.
- **Real robot:** a cheap quadruped (e.g. a Petoi-class kit), where real trials truly are scarce.
- **Plasticity instead of search (F5):** the feedback gains $\kappa$ are updated online by an evolved local rule rather than searched.
