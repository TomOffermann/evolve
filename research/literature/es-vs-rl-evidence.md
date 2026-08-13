# Evidence dossier — is ES actually a viable alternative to GRPO/PPO?

Collected 2026-08-13. Both directions, because the case is contested and the counter-evidence
turns out to be the most useful part.

**One attribution correction first.** OpenAI's ES paper is Salimans et al., *Evolution Strategies
as a Scalable Alternative to Reinforcement Learning* (arXiv:1703.03864, 2017) — foundational, not
new. The recent one that makes the LLM post-training claim is Qiu et al. from **Cognizant AI Lab**,
ICML 2026. Easy to conflate since the titles rhyme; worth keeping straight when citing.

---

## For: the case is stronger than I expected

### 1. Qiu et al., *Evolution Strategies at Scale: LLM Fine-Tuning Beyond RL* (arXiv:2509.24372, ICML 2026)

First full-parameter ES fine-tuning of LLMs at billion scale, no dimensionality reduction.

- **Models:** Qwen-2.5 (0.5B–7B, plus Math-7B and 14B), Llama-3 (1B–8B)
- **Tasks:** Countdown, MATH (OlympiadBench/MATH500/AIME2024/AMC), conciseness, ARC-AGI, Sudoku
- **Baselines:** PPO, GRPO, Dr.GRPO, SimpleRL-Zero, OpenReasoner-Zero
- **Countdown, Qwen-2.5-7B: ES 66.8% vs 57.5% for the best RL baseline**
- **Stability: 15.5× lower standard deviation across runs than GRPO**
- **Reward hacking:** GRPO at β={0, 0.01} produced nonsense symbols to game a conciseness reward.
  ES did not — *without any KL penalty*.

**The hyperparameter result is the one that matters most for us.** ES used a **single fixed
setting — N=30, σ=0.001, α=5e-4 — across every task and every model size**, while the RL baselines
needed a separate sweep per experiment. For anyone with three 1080 Tis rather than a cluster,
"works untuned" is worth more than a few points of accuracy.

Their own stated puzzle: N=30 working across billions of parameters is counterintuitive and
unexplained.

### 2. EGGROLL (arXiv:2511.16652)

- **Countdown, RWKV-7 1.5B: 35% vs GRPO's 23% at equal wall-clock**
- **1024 parallel generations per GPU vs 32 for GRPO**
- 91% of pure inference throughput; ~100× over naive ES

Wall-clock parity is the honest comparison and it is the one they report.

### 3. Salimans et al. 2017 (OpenAI)

The original: ES is competitive with policy gradients on MuJoCo/Atari and parallelises far better
because workers exchange seeds and scalars rather than gradients. Everything since is a scaling
argument on top of this.

---

## Against: and this is the useful part

### 4. *Evolutionary Strategies lead to Catastrophic Forgetting in LLMs* (arXiv:2601.20861)

- **Models:** Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct
- **Target:** Countdown (plus GSM8K/MATH/OlympiadBench). **Held out:** HellaSwag
- **ES loses ~10% on HellaSwag** while converging on the target task. GRPO holds prior-task
  performance while gaining on the new one — ES traces a convex Pareto front, GRPO sits in the
  top-right corner.
- **No mitigation tested, none proposed.** The paper stops at "this remains a challenge."

**The proposed mechanism is what makes this valuable rather than merely discouraging:**

> ES updates have an ℓ₂ norm **three orders of magnitude larger** than GRPO's after 500
> iterations, and **very low sparsity** — GRPO updates are ~95% sparse, ES perturbs nearly every
> parameter. The model drifts far from the base, and that drift is the forgetting.

---

## Why this matters for what we're building

### It is a direct argument for [Proposal B](../proposals/B-modular-partition-diversity.md), and nobody has made it

The diagnosis is *dense, high-norm updates*. Partitioned perturbation makes ES updates **sparse
by construction** — each member perturbs one part of `P`, so the aggregate update touches a
fraction of the network per generation instead of all of it.

That is precisely the property the forgetting paper identifies GRPO as having and ES as lacking.
The paper offers no mitigation; Proposal B is one, it follows from their own mechanism, and it was
motivated here on entirely unrelated grounds (credit assignment and Hamming-distance diversity).

This upgrades P2 from "interesting structural idea" to **"a candidate fix for the strongest
published objection to ES post-training."** It also gives it a second, cleaner success metric
than pass@k: *does partitioned ES retain held-out capability where global ES does not?*

Two things to check before leaning on this: whether partitioning actually reduces update norm
(it may just spread the same total across generations), and whether sparsity per *generation* is
what matters or sparsity of the *cumulative* update — the paper measures the latter.

### It changes the benchmark design

[The benchmark plan](../benchmarks/design.md) measured target-task pass@1/pass@k and reward
hacking. It did **not** measure held-out capability retention — so as specified it could not have
detected the single strongest published objection. Adding HellaSwag (or ARC-easy/PIQA) as a
frozen held-out probe is nearly free, since it is inference-only on a model we already have
loaded.

### It validates the compute plan

Published practice is **N=30**. Our plan assumed N=32–64 at 135M–1.1B. That is not optimistic —
it is what a 14B result was obtained with. The feasibility gate ("does ES move a 135M model at
N=32?") is very likely to pass.

### It sharpens the σ work

Their selling point is that one fixed σ works everywhere while RL needs per-task sweeps. Our
`ResolutionRule` result — that the classical 1/5th rule *inverts* under sparse binary reward, and
that controlling tie rate instead matches a hand-swept σ without sweeping — extends exactly that
argument. If ES's practical edge is "no tuning", removing the last hyperparameter is on the
critical path, not a side quest.

### Numbers to beat, not vibes

| claim | published figure | our benchmark should |
|---|---|---|
| Countdown accuracy | ES 66.8% vs RL 57.5% (Qwen-7B) | reproduce the *direction* at 135M–1.1B |
| Wall-clock parity | 35% vs 23% (RWKV-7 1.5B) | match on one 1080 Ti |
| Stability | 15.5× lower std than GRPO | measure across 8 seeds |
| Reward hacking | ES clean, GRPO hacked without KL | use a hackable reward deliberately |
| Forgetting | ES −10% HellaSwag | **the objection to overturn** |

## Honest caveat about this dossier

Three of the four sources are ES-positive and two share an author community. The forgetting paper
is a single result on two 1-1.5B models with no mitigation attempted, so it is not decisive
either. Neither camp has published a head-to-head where forgetting *and* target performance *and*
wall-clock are all controlled — which is, conveniently, the gap a small careful benchmark can fill
without frontier compute.

**Sources:** [ES at Scale](https://arxiv.org/abs/2509.24372) ·
[EGGROLL](https://arxiv.org/abs/2511.16652) ·
[ES→catastrophic forgetting](https://arxiv.org/pdf/2601.20861) ·
[OpenAI ES 2017](https://openai.com/index/evolution-strategies/) ·
[ES vs GRPO overview](https://www.alphaxiv.org/blog/es-for-fine-tuning-llms)
