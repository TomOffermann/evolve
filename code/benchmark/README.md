# GPU Benchmark — ES vs GRPO/SFT Post-Training

Head-to-head comparison of Evolution Strategies (EGGROLL) against gradient-based
post-training (SFT, GRPO) on a real pretrained language model.

## Quick Start

```bash
pip install -r code/benchmark/requirements.txt
cd code/benchmark

# Quick test (30 gens, ~5 min on one GPU)
python run.py --model HuggingFaceTB/SmolLM2-135M --quick --seeds 0

# Full benchmark (200 gens, 3 seeds)
python run.py --model HuggingFaceTB/SmolLM2-135M --seeds 0,1,2

# Scale demonstration (ES runs where GRPO cannot on 11GB)
python run.py --model Qwen/Qwen2.5-1.5B --methods eggroll_vanilla,eggroll_partitioned,eggroll_selective --seeds 0
```

## Methods

| method | type | what it does |
|---|---|---|
| `sft` | Supervised | Fine-tune on countdown solutions (the supervised ceiling) |
| `grpo` | Gradient RL | Group Relative Policy Optimization (DeepSeek-style) |
| `eggroll_vanilla` | ES | Global IID perturbation + ResolutionRule sigma adaptation |
| `eggroll_partitioned` | ES | Partitioned perturbation: each member perturbs one layer-group |
| `eggroll_selective` | ES | Partitioned + utility-gated update: only update useful parts |

## What We Measure

- **pass@1, pass@4, pass@16** — target task accuracy and diversity
- **Perplexity delta** — forgetting probe. ES is known to cause ~10% HellaSwag
  degradation (arXiv:2601.20861). Partitioned ES should close this gap by making
  updates sparse by construction.
- **Wall-clock time** — the practitioner's question
- **Per-part utility** — which layers of the model are being updated (ES only)

## The Hypothesis

The strongest published objection to ES post-training is catastrophic forgetting:
dense, high-norm updates drift the model far from the pretrained base. GRPO updates
are ~95% sparse; ES updates touch every parameter.

**Partitioned ES makes updates sparse by construction.** Each member perturbs one
layer-group of P, so the aggregate update only touches a fraction of the network
per generation. The selective variant goes further: it only applies updates to
parts that showed positive utility, leaving the rest frozen.

This is the mechanism arXiv:2601.20861 identifies as missing, and it was motivated
here independently (on credit-assignment grounds from DEGA/DiPEC).

## ES Hyperparameters

Default from Qiu et al. (arXiv:2509.24372, ICML 2026):
- N=32, sigma=0.001, alpha=5e-4, rank=1
- These work untuned across tasks and model sizes up to 14B
- ResolutionRule adapts sigma automatically (targets 50% tie rate)

## Hardware Requirements

- **SmolLM2-135M:** Any GPU with 4+ GB. ~7 min per ES run (200 gens).
  GRPO fits too, so head-to-head comparison is possible.
- **Qwen2.5-1.5B:** Any GPU with 8+ GB for ES (inference only).
  GRPO needs 18+ GB (weights + gradients + optimizer). This is the structural
  advantage: ES runs where gradient methods cannot.
