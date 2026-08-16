"""Benchmark configuration. Dataclass-based, not argparse spaghetti.

Default hyperparameters from Qiu et al. (arXiv:2509.24372, ICML 2026):
N=30, sigma=0.001, alpha=5e-4 — works untuned across tasks and model sizes
up to 14B. We use N=32 for a power-of-two that aligns with GPU batch sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    # Model
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    dtype: str = "float32"           # float32 or bfloat16 (NOT float16 on Pascal)
    device: str = "cuda"

    # Task
    task: str = "countdown"          # "countdown" or "gsm8k"
    n_problems_train: int = 32       # problems per ES generation (CRN batch)
    n_problems_eval: int = 256       # held-out evaluation set
    max_new_tokens: int = 24         # 24 for countdown, 256 for GSM8K

    # ES hyperparameters (from Qiu et al.)
    n_pop: int = 32
    rank: int = 1
    sigma: float = 0.001
    alpha: float = 5e-4
    generations: int = 200

    # Partitioning
    partition_groups: int | None = None  # None = auto (ceil(n_layers / 4))
    n_active: int = 1                    # parts per member for PartitionedSampler

    # Evaluation
    eval_every: int = 25             # generations between evaluations
    eval_ks: list[int] = field(default_factory=lambda: [1, 4, 16])
    temperature: float = 0.7         # for pass@k sampling

    # Run
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    methods: list[str] = field(default_factory=lambda: [
        "sft", "grpo", "eggroll_vanilla", "eggroll_partitioned", "eggroll_selective",
    ])
    output_dir: str = "code/benchmark/results"

    # Forgetting probe
    forgetting_tasks: list[str] = field(default_factory=lambda: ["hellaswag"])

    def __post_init__(self):
        if self.task == "gsm8k" and self.max_new_tokens < 128:
            self.max_new_tokens = 256
