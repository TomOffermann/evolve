"""JSON logging and result collection for the GPU benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    method: str
    model: str
    task: str
    seed: int
    config: dict
    checkpoints: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    forgetting: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    wall_seconds: float = 0.0
    method_info: dict = field(default_factory=dict)

    def save(self, output_dir: str):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        fname = path / f"{self.method}_{self.model.split('/')[-1]}_seed{self.seed}.json"
        with open(fname, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return fname


def load_results(results_dir: str) -> list[RunResult]:
    results = []
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        results.append(RunResult(**{k: v for k, v in data.items()
                                    if k in RunResult.__dataclass_fields__}))
    return results
