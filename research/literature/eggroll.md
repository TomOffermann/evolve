# EGGROLL — Evolution Strategies at the Hyperscale

**Paper:** Sarkar, Duque, Fellows, Foerster et al., *Evolution Strategies at the Hyperscale*,
arXiv:2511.16652. Oxford FLAIR/FORL + Mila + NVIDIA. Site: <https://eshyperscale.github.io/>
**Status:** Settled as our optimizer substrate (see `claude/decisions/0001-substrate-and-scope.md`).

## What it does

Standard ES perturbs every weight matrix with a dense Gaussian `E ∈ R^{m×n}`. That is fine
mathematically and catastrophic on a GPU: every population member needs its own full-rank
matmul, so batched ES runs at roughly **1% of pure inference throughput** on an H100. The
bottleneck is memory bandwidth, not FLOPs.

EGGROLL replaces the dense perturbation with a **rank-r** one:

```
A_i ∈ R^{m×r},  B_i ∈ R^{n×r}   iid, zero-mean, unit-variance
E_i = (1/√r) A_i B_iᵀ            (1/√r keeps Var[E] bounded as r grows)
W_i = W + σ E_i
```

The forward pass then factorises so the shared part is computed **once** for the whole population:

```
y_i = x Wᵀ + σ (x B_i) A_iᵀ
```

With `r = 1` the per-member work collapses to vector ops. Reported: **91% of pure batch
inference throughput**, ~**100× faster** than naive ES at billion-parameter scale.

## The update

```
ΔW = (α / (N σ)) · Σ_i F̃_i · A_i B_iᵀ ,     F̃_i = F_i − F̄
```

**Key point:** individual perturbations are rank-r, but the *sum of N of them* has rank up to
`min(N·r, m, n)`. So with `N ~ 10^6` the update is full-rank. This is the difference between
EGGROLL and "just run ES on a LoRA adapter" — LoRA permanently constrains the update to low
rank; EGGROLL uses low rank only as a *computational device*. That distinction matters for
pretraining, less so for finetuning.

## Facts that constrain everything we build on top

1. **Members are seeds.** Noise is regenerated on demand from a counter-based RNG
   (`jax.random.fold_in`); `A_i, B_i` are never stored. Storage per member is O(1), not O(mn).
2. **N is enormous.** Experiments span N = 2 … 2^20 (1,048,576). ~1024 members per GPU in LLM
   finetuning. Anything O(N²) is dead above N ≈ 10^3.
3. **No antithetic sampling.** The paper samples independently, no mirrored pairs. I originally
   wrote this up as "a free control variate they left on the table". **Probably wrong** —
   [E1](../experiments/E1-results.md) measured mirrored pairs as *worse* than independent
   sampling (−0.18 mean). At a fixed evaluation budget, mirroring buys variance reduction by
   halving the number of *independent* exploration directions, N → N/2, and in high dimension the
   direction count dominates. That is a plausible reason the authors sample independently.
   *(Provisional: that measurement comes from the E1 run that carried an RNG-ordering flaw. The
   margin is large relative to typical standard errors, but it deserves a clean rerun.)*
4. **`r = 1` is the workhorse.** Convergence is already fast at minimal rank.
5. **Inference-only.** No backward pass, no activation storage, no optimizer state.
   ~half the memory of Adam training. This is what makes weak hardware viable (see
   [P3](../problems/P3-objectives-and-hardware.md)).
6. **`(x B_i) A_iᵀ` is computed anyway.** That per-member activation delta is a free,
   rank-r functional signature of the member. We should exploit this — see
   [Proposal A](../proposals/A-functional-signature-diversity.md).

## Evaluated on

Pure-integer RNN LM pretraining (character-level, minipile) · tabula-rasa RL across 16 envs
(Navix, Craftax, Brax, Kinetix, Jumanji) · LLM reasoning finetuning vs GRPO (countdown, GSM8K,
RWKV-7) · integer-quantised distillation · a high-frequency-trading time-series model.

## Stated limits / our reading

- Large N is *critical* for pretraining; two-point zeroth-order methods don't cut it there.
- Integer-quantised finetuning still needed Adam.
- Authors flag neurosymbolic / non-differentiable pipelines as the natural next target.
  We agree, and would add self-supervised latent objectives — see
  [P3](../problems/P3-objectives-and-hardware.md).

## Related

- `eggroll-es` on PyPI (v0.2.0) — check licence and API before reimplementing.
- Qiu et al., *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*,
  arXiv:2509.24372 — full-parameter ES on LLMs, dense perturbations, seed-based noise retrieval,
  layer-level in-place perturbation. Reports better tolerance to delayed reward and less reward
  hacking than RL. Different point in the design space, same family. Code:
  <https://github.com/VsonicV/es-fine-tuning-paper>
