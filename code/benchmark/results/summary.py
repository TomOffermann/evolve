"""Comparison tables from benchmark results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def print_comparison(results_dir: str):
    """Print comparison table across methods and seeds."""
    by_method = defaultdict(list)
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fp:
            r = json.load(fp)
        by_method[r["method"]].append(r)

    if not by_method:
        print("No results found.")
        return

    method_order = ["sft", "grpo", "eggroll_vanilla", "eggroll_partitioned",
                    "eggroll_selective"]
    methods = [m for m in method_order if m in by_method]
    for m in sorted(by_method):
        if m not in methods:
            methods.append(m)

    print(f"\n{'method':25s} {'seeds':>5s} {'pass@1':>8s} {'sem':>6s} "
          f"{'pass@16':>8s} {'sem':>6s} {'ppl_Δ':>7s} {'wall_s':>7s}")
    print("-" * 80)

    for method in methods:
        runs = by_method[method]
        n = len(runs)

        p1 = np.array([r["final"].get("pass@1", 0) for r in runs])
        p16 = np.array([r["final"].get("pass@16", 0) for r in runs])
        ppl_delta = np.array([r.get("forgetting", {}).get("perplexity_delta", 0)
                              for r in runs])
        wall = np.array([r["wall_seconds"] for r in runs])

        def sem(x):
            return x.std(ddof=1) / n**0.5 if n > 1 else 0

        print(f"{method:25s} {n:>5d} {p1.mean():>8.4f} {sem(p1):>6.3f} "
              f"{p16.mean():>8.4f} {sem(p16):>6.3f} "
              f"{ppl_delta.mean():>+7.1f} {wall.mean():>7.0f}")
