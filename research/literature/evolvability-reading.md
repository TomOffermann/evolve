# Reading notes: the three papers closest to "Evolvable Representations"

*For the proposal `research/evolvable/evolvable_representations.tex`. Each entry covers: what the paper does, what it tries to achieve, how it differs from our setting, and how to answer "isn't this just X?".*

PDFs (download manually; both the cloud and your computer's sandbox block arXiv/PNAS downloads, so run this in your own terminal):

```bash
mkdir -p ~/Code/evolve/research/literature/papers && cd ~/Code/evolve/research/literature/papers
curl -L -o gajewski2019_evolvability_es.pdf https://arxiv.org/pdf/1907.06077
curl -L -o raghu2020_anil.pdf            https://arxiv.org/pdf/1909.09157
# PNAS (open access); if curl gets a 403, open it in the browser and save as PDF:
open https://www.pnas.org/doi/10.1073/pnas.0503610102
```

---

## 1. Gajewski, Clune, Stanley, Lehman — *Evolvability ES: Scalable and Direct Optimization of Evolvability* (GECCO 2019, arXiv:1907.06077)

### Main idea
- **Definition.** Evolvability is the *ability to further adapt*, operationalised as how **diverse the behaviours** of a solution's random mutants are.
- **Method.** An ES keeps a Gaussian search distribution over network weights, $z\sim\pi(\cdot;\theta)$ (mean = the "central individual"). Instead of maximising expected reward, it maximises the **behavioural diversity of the population it samples**. Two variants:
  - **MaxVar:** $J(\theta)=\sum_j \mathbb{E}_z[(B_j(z)-\mu_j)^2]$, the trace of the covariance of the behaviour characterisation $B(z)$.
  - **MaxEnt:** $J(\theta) = -\mathbb{E}_z[\log \mathbb{E}_{z'}\,\varphi(B(z')-B(z))]$, the entropy of the behaviour distribution under a kernel density estimate.
- **Gradients** come from the same samples through (nested) stochastic computation graphs. Diversity information is shared between population members, so no extra rollouts are needed. This is what makes it scale, compared with earlier evolvability search that evaluated many offspring per individual.
- **Experiments.** Planar Half-Cheetah and 3-D Ant, policies with about 75k parameters, population 10,000. $B$ is the final (x) or (x, y) position of the robot.
  - The mutants of an evolvable policy walk in *all* directions.
  - A single mutation step moves them almost as far as policies trained for distance: 92–94 % (2-D) and 83–85 % (3-D) of standard ES distance.
  - Compared with MAML (after the same random mutations), the mutants move further and are more diverse (p < 0.00005).
  - Seeding standard ES with an evolvable policy adapts faster to new directions (p < 0.0005 after 5, 10 and 20 generations).

### What it tries to achieve
A *parameter vector* (a policy) from which **random mutation** reaches many different behaviours. Evolvability is optimised **directly and at scale**, as a property of a *single solution plus its mutation distribution*.

### How it differs from our setting

| | Evolvability ES | Ours |
|---|---|---|
| What is shaped | the policy's weights, i.e. where the search *starts* | the **representation** (encoder) on which a *separate, tiny* genome is searched |
| Evolvability measure | behavioural **diversity** of mutants (variance / entropy of $B$) | **evaluations to adapt** to related tasks: epistasis + task distance |
| Search that uses it | the same continuous ES over all ~75k weights | a discrete (1+1) EA over $n\approx64$ ternary votes, reward only |
| Theory | none (empirical) | lemma: epistasis ⇔ co-activation near threshold; $O(n\log k)$ re-adaptation |
| Forgetting | adapting changes all weights | encoder frozen at deployment; genomes stored per task |

**Answer to "isn't this Evolvability ES?"** No. Diversity of mutants is neither necessary nor sufficient for fast adaptation to a *specific* new task. A landscape can be highly diverse and still rugged. We target the landscape's *structure* (additivity, small $k$), which is what runtime theory ties to adaptation cost. Their objective could be an extra **baseline** for us: train the encoder so that random genomes produce diverse task predictions.

---

## 2. Raghu, Raghu, Bengio, Vinyals — *Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML* (ICLR 2020, arXiv:1909.09157)

### Main idea
- **Question.** MAML learns an initialisation from which a few gradient steps solve a new task. Does it work because of **rapid learning** (large, efficient changes in the inner loop) or **feature reuse** (the initialisation already contains good features and the inner loop barely changes them)?
- **Tests.**
  - Freeze parts of the network during the inner loop.
  - Measure how much the representations change during adaptation (CCA/CKA similarity before and after the inner loop).
- **Finding: feature reuse dominates.** The body's representations barely change in the inner loop; essentially only the head adapts.
- **ANIL (Almost No Inner Loop).** Run the inner loop *only on the final layer (head)*, in both meta-training and testing. It matches MAML on few-shot image classification (Omniglot, MiniImageNet) and on RL benchmarks, at a fraction of the cost.
- **NIL (No Inner Loop) at test time.** Remove the head and classify by similarity of the learned features to the support examples. It works about as well, so test performance is determined almost entirely by **feature quality**.

### What it tries to achieve
To explain *why* gradient-based meta-learning works. The answer: meta-learning is mostly **representation learning for a fast linear/head adaptation**.

### How it differs from our setting

| | ANIL | Ours |
|---|---|---|
| Fast learner | gradient descent on a continuous head, **with labels** | EA on a discrete ternary head, **reward only** |
| Slow learner | encoder, meta-trained through the (differentiable) inner loop | encoder, trained by gradients **at the genome the EA found** (first-order, no backprop through the EA) |
| What "good features" means | features on which a *gradient step* on the head separates classes | features on which the *EA's landscape* over the head is near-additive with small task distance |
| Structure of fast weights | continuous, dense | discrete, sparse (parsimony), stored per task |

**Why it matters for us.** ANIL is the **direct gradient analogue** of our architecture: slow encoder plus fast head. Its central finding (feature quality determines adaptation) is exactly our premise, stated for gradients. Our first Stage-1 result says the same thing more strongly for EAs: on autoencoder features CMA-ES still adapts, but the discrete EA collapses (2585 vs 434 evaluations). This makes a clean framing: *"ANIL showed that meta-learning is feature learning for a gradient head; we ask what feature learning for an evolutionary head looks like, and show that the answer differs."*

**Answer to "isn't this ANIL with an EA?"** The inner learner changes the objective the features should satisfy. A gradient head needs linearly separable features. A ternary EA head needs **low co-activation near the threshold and redundancy for margins** (our lemma), and it works from scalar rewards without labels. The weighted-task experiment below is designed to show where the two requirements come apart.

---

## 3. Kashtan & Alon — *Spontaneous evolution of modularity and network motifs* (PNAS 102(39), 2005)

### Main idea
- **Model.** Boolean circuits of NAND gates (4 inputs, 1 output, fewer than about 12 gates), evolved by a GA with a population of about 1000.
- **Protocol: fixed goal vs *modularly varying goals* (MVG).** The target alternates every 20 generations between two functions that share sub-problems:
  $G_1=(a \oplus b)\wedge(c\oplus d)$ and $G_2=(a\oplus b)\vee(c\oplus d)$.
- **Findings.**
  - Under a **fixed goal**, evolved circuits are *non-modular* (modularity $Q\approx0.12$) and take on the order of $10^4$ generations. Even modular starting populations lose their modularity.
  - Under **MVG**, circuits become **modular** ($Q\approx0.54$): the shared XOR modules are reused and only the combining gate changes. After a goal switch the population re-adapts in **about 5 generations**, needing only about 2 rewiring changes.
  - MVG also finds perfect solutions **faster overall** than a fixed goal, and produces recurring network motifs.
  - If the varying goals share no modular structure, no modularity emerges.
- *(Numbers taken from lecture notes summarising the paper, CMU 02-251; check against the PDF before citing.)*

### What it tries to achieve
To explain the *origin of modularity* in biological networks. Modularity is not selected for directly; it emerges because environments vary in a modular way, and modular solutions re-adapt fastest.

### How it differs from our setting

| | Kashtan & Alon | Ours |
|---|---|---|
| What evolves | the whole circuit (structure and function) | only a small genome; the representation is trained by gradients |
| Source of evolvability | environmental variation (MVG) acting on evolution itself | an explicit training objective (evolution in the loop plus epistasis penalties) |
| Evidence | simulation | runtime theory plus experiments |

**Why it matters for us: it is the biological version of our small-$k$ claim.** Under MVG, related goals differ in about 2 "genome" changes, so re-adaptation is fast. That is precisely $k_{\tau\tau'}$ small, which with $O(n\log k)$ is what we want a representation to provide. Our task family is already MVG-like: successors change one factor or one sign. The first Stage-1 run showed the EA finds **dense** genomes ($k\approx25$), which destroys this. With the new parsimony tie-break, oracle genomes shrink from 42 to 11.5 non-zeros (sanity check 10). That should let related tasks sit at small $k$, which is the Kashtan–Alon regime.

**Answer to "biology showed this already".** They showed that modularity *emerges* under varying goals, in a GA on tiny circuits, with no guarantee. We (i) *train* for it in a deep encoder, (ii) quantify it (epistasis, $k$), and (iii) connect it to runtime bounds.

---

## Summary: where our proposal sits

- **Evolvability ES:** makes *solutions* evolvable (diverse mutants), with no theory, and adaptation still searches all weights.
- **ANIL:** shows fast adaptation equals *good features for a gradient head*.
- **Kashtan & Alon:** varying goals make *evolved structure* modular, and modular structure re-adapts in few changes.
- **Ours:** trains *features for an evolutionary head*, so that the EA's landscape is near-additive with small task distance. That links representation learning to EA runtime theory, from reward alone and without forgetting.

## Experiments added because of this reading (code: `stage1_evolvable.py`)

| flag | what it tests | link |
|---|---|---|
| `--set parsimony=true` | Genomes stay sparse (lexicographic reward, then fewer votes); $k$ should drop and transitions should speed up | Kashtan–Alon small-$k$ regime |
| `--set max_weight=2` | Tasks need weight-2 votes. A gradient head just uses a larger weight; a ternary EA head needs **redundant features**. Prediction: multitask features become poor for the EA, while EIL with the margin term learns redundancy | where ANIL-style and EA-style feature requirements differ |
| `--set split=size` | Meta-train on 3-factor rules, evaluate on 5-factor rules | generalisation beyond the training task family |
