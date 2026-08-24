# MoE Routing Experiment

Can ES optimise Mixture-of-Experts routing decisions better than gradient-based methods?

## Motivation

MoE routing is inherently discrete: an argmax selects which expert processes each token. Gradient methods cannot differentiate through argmax, so they relax it to a softmax-weighted sum during training and hope the soft routing transfers to hard routing at inference. ES evaluates the hard routing directly and does not need the relaxation.

This experiment tests whether that advantage is real.

## Setup

1. A toy MoE model (~1M params) with K=4 expert FFN modules, a shared embedding, a linear router, and a shared output head.

2. Task: modular arithmetic -- predict `sum(tokens) mod 16`. Simple enough for experts to learn, structured enough for different experts to specialise.

3. Pre-train the full model with SGD (soft routing) so the experts learn useful features.

4. Reset the router to near-uniform, then optimise it three ways:

   - **Gradient (Adam)**: Cross-entropy loss with soft routing. The standard approach.
   - **ES global**: EGGROLL-style low-rank perturbation of the full router weight matrix.
   - **ES partitioned**: Each ES member perturbs routing to a single expert. Gives per-expert credit assignment via PartitionedSampler.

## Metrics

- **Accuracy**: Fraction of validation examples classified correctly under hard routing.
- **Expert utilization**: Fraction of tokens sent to each expert.
- **Load balance**: Entropy of routing distribution, normalised to [0, 1]. 1.0 = perfectly uniform.
- **Collapsed experts**: Number of experts receiving < 1% of traffic.
- **Wall-clock time**.

## Run

```bash
python code/experiments/moe_routing/run.py
```

Runs on CPU in under 5 minutes. No GPU required.

## Connection to EGGROLL

The experiment uses the evolve framework directly:
- `Trainer` and `TrainConfig` from `evolve.core.trainer`
- `IIDSampler` and `PartitionedSampler` from `evolve.operators.sampling`
- `RankWeighting` from `evolve.operators.weighting`
- The `Objective` protocol from `evolve.core.types`

The partitioned arm is a direct application of Proposal B (validated in E8): each member perturbs one part of the parameter space, giving per-part credit assignment. Here "parts" are expert routing rows, so the credit assignment is per-expert utility.
