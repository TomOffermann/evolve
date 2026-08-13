# Benchmark design — what would make "ES instead of GRPO/PPO" a real claim

**Status:** Open, proposed 2026-08-13. Nothing here is built yet.

Constraint: **a few GTX 1080 Tis.** 11 GB, no tensor cores, fp16 crippled to 1/64 of fp32, int8
DP4A at ~4× fp32. Assume ~3 TFLOPS effective for small models. Every number below is computed
from those, not guessed.

**Literature grounding:** [evidence dossier](../literature/es-vs-rl-evidence.md) — the published
numbers to beat, and the one objection this benchmark must be able to detect.

## The bar

Everything measured so far is on a 94k-parameter from-scratch policy. That is enough to test
*mechanisms* against each other and nothing else. To claim ES is an alternative to GRPO/PPO, a
benchmark has to have all five:

1. **A real pretrained base.** Post-training means eliciting and reshaping existing capability.
   A from-scratch toy has none, so nothing about it transfers.
2. **A head-to-head baseline at matched compute**, on the same hardware. Without GRPO in the same
   table, "ES works" is unanchored.
3. **Verifiable reward.** The RLVR setting, where the comparison is actually contested.
4. **pass@k, not just pass@1.** The axis where ES might genuinely differ, and the documented
   failure of GRPO.
5. **A stated way to lose.** Given this project's record, assume the first positive result is an
   artefact until its control has been audited.

## Benchmark A — RLVR post-training vs GRPO

### The task pair, and why it is a *pair*

This is the part worth getting right, because it builds the control into the benchmark rather
than bolting one on afterwards.

| task | solutions per problem | diversity should |
|---|---|---|
| **Countdown** (4 numbers → target) | many | **matter** — pass@k is real |
| **Multi-digit arithmetic** (exact answer) | exactly one | **not matter** — pass@k is capped |

A diversity mechanism that helps on both is a generic regulariser and we have learned to
disbelieve those. A mechanism that helps on countdown and is *neutral* on arithmetic is doing
what it claims. **The second task is a control at the task level** — the thing every overturned
result in this project was missing.

Countdown also gives continuity: EGGROLL itself used it, so results are comparable against a
published claim, and our 94k toy version is the same task shape.

### Model

**SmolLM2-135M** for the main sweep, **Qwen2.5-0.5B** for a scale check.

| model | ES memory (fp32) | GRPO memory (weights+grad+Adam) | ES s/gen (N=32,B=32,24 tok) |
|---|---|---|---|
| SmolLM2-135M | 0.54 GB | 2.2 GB | **2.2 s** |
| SmolLM2-360M | 1.45 GB | 5.8 GB | 5.9 s |
| Qwen2.5-0.5B | 1.98 GB | 7.9 GB | 8.1 s |

200 generations at 135M is **~7 minutes per run**. Eight seeds on three cards is under half an
hour. This is comfortably affordable, which means we can afford the seed counts that E5/E6
lacked.

**KV cache is the binding memory constraint, not weights** — 5.7 GB at N=32, B=32. The framework
already chunks the population, so cap the chunk at ~16 members and memory stops mattering.

### The GRPO baseline

Non-negotiable, and it must run on the *same card*. TRL's GRPOTrainer on 135M fits easily.
Match on **wall-clock on one 1080 Ti**, and report forward-equivalent FLOPs alongside — wall-clock
is what a practitioner cares about, FLOPs is what makes it comparable elsewhere.

Fair-budget accounting: GRPO does G rollouts per prompt plus a backward pass (~2× forward); ES
does N forward-only rollouts. At G=8 and N=32, ES uses ~4× the rollouts and no backward, so the
two land closer than they look. Report the ratio explicitly rather than asserting parity.

### Metrics

- **Held-out capability retention** — a frozen probe (HellaSwag, ARC-easy, PIQA) evaluated before
  and after. **Added 2026-08-13 after the literature pass**: arXiv:2601.20861 reports ES losing
  ~10% on HellaSwag where GRPO loses nothing, and identifies dense high-norm updates as the cause.
  As originally specified this benchmark **could not have detected the strongest published
  objection to ES post-training**. The probe is inference-only on a model already loaded, so it is
  nearly free. See the [evidence dossier](../literature/es-vs-rl-evidence.md).
- **pass@1, pass@4, pass@16** — the diversity axis
- **reward-hacking rate** — fraction of accepted outputs that game the verifier (degenerate
  expressions, format exploits). EGGROLL claims ES hacks less; this is where to check
- **stability** — cross-seed variance and divergence rate, at 8+ seeds
- **wall-clock to a fixed pass@1 threshold** — the practitioner's question

## The hardware demonstration — and this one changed after doing the arithmetic

I had assumed the "ES trains bigger models on weak cards" claim would show up at 135M–500M. It
does not: **GRPO fits comfortably at all three sizes above.** The claim is real but it bites
later, and pretending otherwise would have produced a benchmark that quietly proves nothing.

The crossover is at **~700M parameters**, where `4 × params` (weights + grads + Adam m,v) passes
11 GB. So the clean demonstration is:

> **TinyLlama-1.1B or Qwen2.5-1.5B post-trained with ES on a single 11 GB card, where GRPO cannot
> run at all.** ES needs ~4.4 GB of weights plus a chunked KV cache (~6 GB total). GRPO needs
> ~17.6 GB, and even 8-bit Adam leaves it borderline once activations are counted.

Cost: ~9 s/generation at N=32, B=16, 24 tokens → **~30 minutes for a 200-generation run.**
Entirely affordable, and it is an unarguable structural claim rather than a contested one.

This should be built *after* Benchmark A works at 135M, but it is the single most persuasive
artefact available on this hardware.

## Benchmark B — JEPA, scoped honestly

**ES pretraining a JEPA from scratch is out of reach here and should not be attempted.** ES
gradient-estimate quality degrades with parameter count, EGGROLL's answer is N ~ 10⁶, and we have
three cards. Claiming otherwise would burn weeks to reproduce a known limitation.

The defensible, feasible claim is the one [P3](../problems/P3-objectives-and-hardware.md) actually
argues:

> **ES can optimise objective terms SGD cannot** — non-differentiable ones — *on top of* what SGD
> already gives you.

### Protocol

1. Pretrain a small encoder (~5–20M param ViT, CIFAR-10, patch 4 → 64 tokens) with **SGD** and
   the standard I-JEPA machinery (EMA target, stop-gradient). This is the base.
2. **ES-finetune** it against an objective that *adds* non-differentiable terms: latent effective
   rank (SVD + entropy), retrieval hit-rate on held-out pairs, discrete codebook occupancy.
3. Compare against SGD finetuning with the closest differentiable surrogates (VICReg
   variance/covariance).

**Metric: linear-probe accuracy on CIFAR-10 labels**, the standard SSL protocol and one a
collapsed encoder cannot fake. Plus the non-differentiable target itself.

Cost: ~1.7 s/generation at N=32, batch 256. A 1000-generation run is under an hour.

E3 already showed the non-differentiable rank bonus beating every differentiable surrogate at its
own target on pendulum. This is that result on a benchmark somebody else would recognise.

Keep pendulum as the cheap unit test — it runs in seconds and catches regressions.

## Compute plan, three cards

- **fp32 or int8. Never fp16.** Gaming Pascal runs fp16 at 1/64 of fp32; a naive bf16 port will
  look like the algorithm being slow.
- Cards run **independent seeds**, not a sharded population — seeds are the scarce resource given
  how many results here died of low power, and independent runs need no interconnect at all.
- Chunk the population to ~16 members so KV cache never binds.
- int8 via DP4A is a ~4× win and EGGROLL already demonstrated integer-only training, but it is
  real kernel work. Defer until fp32 results exist.

Rough budget for the full Benchmark A sweep: 2 tasks × 4 arms × 8 seeds × 7 min ≈ **7 hours of
card time**, ~2.5 hours wall-clock across three cards.

## Feasibility gates — run these before committing

Two cheap checks, either of which kills the plan early:

1. **Does ES move a 135M model at all?** N=32 over 135M parameters is a very small number of
   search directions. Gate: 30 generations, does training reward rise above the SFT base at all?
   *Prior: very likely yes.* Qiu et al. obtained their Qwen-14B results at **N=30 with a single
   fixed σ across every task and model size** — our N is not optimistic, it is standard practice.
2. **Does the base model have headroom?** Countdown pass@1 for SmolLM2-135M must be nonzero and
   well under ceiling. If it is 0%, ES has no signal to amplify; if it is 60%, there is nothing to
   show. Target 5–20%.

## What is *not* testable at this budget — state it rather than imply it

- Whether any of this holds at 7B+. Nothing here will speak to frontier post-training.
- Long-horizon / delayed-reward advantages (EGGROLL's claim 2) — needs multi-turn agentic tasks
  well beyond 24-token completions.
- ES pretraining of anything.
- Whether ES beats GRPO on *pass@1*. The plausible claim is comparable pass@1 at better pass@k,
  and the benchmark should be framed to test that rather than a stronger one we cannot support.
