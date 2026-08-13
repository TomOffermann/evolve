# P3 — Objectives beyond RL, and hardware we can actually get

**Status:** Open. Tom's idea, 2026-08-12.

## Part 1: EGGROLL on non-RL objectives

EGGROLL is presented on RL, LM pretraining, reasoning finetuning and a trading model. Nothing in
the method is RL-specific — it needs a scalar fitness per member and nothing else. That opens up
objectives that gradient descent handles badly or not at all.

### JEPA is an unusually good fit, and here's the non-obvious reason

Joint-Embedding Predictive Architectures predict latent embeddings of masked regions from visible
context. Their central failure mode is **representation collapse**: the encoder can drive the loss
to zero by outputting a constant. The entire apparatus of stop-gradients, momentum/EMA target
encoders, and architectural asymmetry exists to stop gradient descent from finding that shortcut.
VICReg-style variance/covariance penalties are the explicit version of the same fix.

With ES you don't have that problem in the same shape. Fitness is a black box. You can put an
anti-collapse term **directly in the fitness** — effective rank of the embedding batch, minimum
singular value, per-dimension variance floor, a retrieval hit-rate on held-out pairs — including
terms that are non-differentiable, rank-based, or discrete, which you cannot backprop through.

**The research claim worth testing:** the asymmetry hacks (stop-gradient, EMA target) are
artefacts of *how we optimise*, not of the objective. Remove backprop and you may be able to
remove them, and just say what you want: "embeddings must predict, and must not collapse."

> **Tested — [E3](../experiments/E3-results.md), 2026-08-13.** Built and run on pendulum, no
> stop-gradient or EMA target in any arm.
>
> **Supported in part.** A non-differentiable objective term — the effective rank of the latent
> batch, an SVD followed by an entropy — is optimisable by ES and achieves the highest latent
> rank (6.72 of 8) versus 2.48–5.24 for the differentiable surrogates. That is the concrete
> version of "state what you want instead of a differentiable proxy for it".
>
> **Not demonstrated.** The prediction-only control scale-collapsed (latent std 0.078 vs 1.20)
> but never *information*-collapsed — probe R² was flat at 0.72 across all arms. The control did
> not reach the failure the hacks exist to prevent, so the strong claim is unproven. Needs a
> harder prediction problem or a higher-capacity encoder.
>
> Also learned the hard way: normalising prediction error by latent variance is *itself* an
> anti-collapse mechanism, and a fully-observed pendulum makes prediction so easy that collapse
> is never the attractor. Both accidentally prevented the control from failing.

Cheap first version: small JEPA, small dataset, ES-finetune a pretrained encoder rather than
pretraining from scratch. Full ES pretraining of a JEPA is a large ask — EGGROLL itself notes
that pretraining is where huge `N` becomes critical.

### Other objectives ES unlocks

Non-differentiable metrics as first-class objectives (exact-match, pass@k, retrieval recall,
compiler/verifier output), discrete latent codebooks without straight-through estimators,
symbolic/neurosymbolic pipelines with a non-differentiable component in the middle. The EGGROLL
authors flag this direction themselves.

## Part 2: Hardware — why weak GPUs are a good fit, not a compromise

**The structural argument.** ES workers exchange `(seed, fitness)`. That is a few bytes per member
per generation. Data-parallel SGD exchanges gradients — gigabytes per step. A heterogeneous pile
of old GPUs on a mediocre interconnect is close to the *worst* case for SGD and close to the
*best* case for ES. And ES is inference-only: no backward, no activations, no optimizer state,
roughly half the memory of Adam training. This is not "we'll make do with 1080 Tis" — the method
is genuinely well-matched to that hardware.

**The Pascal gotcha, in advance, because it will cost a week otherwise.**
On GTX 1080 Ti (GP102, sm_61) there are no tensor cores and **fp16 throughput is 1/64 of fp32** —
NVIDIA crippled it deliberately on gaming Pascal. Naively porting a bf16/fp16 pipeline to a 1080 Ti
will be catastrophically slow and it will look like the algorithm's fault.

What Pascal *does* have is **DP4A**: 4-way 8-bit integer dot product with 32-bit accumulate, at
roughly **4× fp32 throughput**. And EGGROLL already demonstrated pure-integer RNN LM pretraining
and integer-quantised distillation.

So the 1080 Ti path is: **fp32 or int8. Never fp16.** Which happens to line up with the part of
EGGROLL that is most novel anyway.

### Target ladder

| Tier | Hardware | Purpose | Notes |
|---|---|---|---|
| 0 | MacBook (MPS / MLX) | Debug, unit tests, toy tasks | Small `N`. Correctness, not speed. |
| 1 | Single 3090 / 4090 | Real experiments | 24 GB, bf16 + tensor cores. The main dev target. |
| 2 | ETH cluster, n × 1080 Ti | Scale-out | fp32/int8 only. Seeds+fitness over the wire. Plentiful availability is the point. |
| 3 | Mixed 3090 + 1080 Ti | Heterogeneous swarm | ES tolerates stragglers well; async generations are viable. |

## Part 3: build or reuse

`eggroll-es` exists on PyPI (v0.2.0), JAX-based. Open question: reuse it, or write our own
GPU-optimised implementation?

Argument for our own: [P2](P2-structured-perturbation.md) needs partitioned masks and per-part
bookkeeping deep inside the perturbation path, and int8 DP4A kernels for Pascal are certainly not
in there. Argument for reuse: getting the counter-based RNG, the fused low-rank forward, and the
distributed loop right is real work that isn't the research.

**Recommendation (Open):** reuse `eggroll-es` as the reference implementation and correctness
oracle; write our own perturbation/aggregation layer against a small interface so we can swap in
partitioned masks and int8 kernels without forking the whole thing. Check its licence first.
