"""Aggregate CPU benchmark results across seeds.

Reads all JSON files from a results directory and prints:
1. Mean/sem table across seeds
2. Pairwise delta/sem significance vs the baseline
3. Checkpoint trajectories (pass@1 and pass@16 over training)

Usage:
    python code/cpu_benchmark/aggregate.py results/
    python code/cpu_benchmark/aggregate.py results/ --csv   # CSV output
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir):
    """Load all JSON result files, grouped by arm name."""
    results = defaultdict(list)
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fp:
            r = json.load(fp)
        results[r["arm"]].append(r)
    return dict(results)


def print_table(results, csv=False):
    """Print comparison table with mean, sem, and delta vs baseline."""
    # Baseline is the first arm (iid_fixed_swept if present, else first alphabetically)
    arm_order = ["iid_fixed_swept", "iid_resolution", "partitioned_resolution",
                 "partitioned_selective", "novelty_resolution", "orthogonal_resolution"]
    arms = [a for a in arm_order if a in results]
    # Add any arms not in the expected order
    for a in sorted(results):
        if a not in arms:
            arms.append(a)

    if not arms:
        print("No results found.")
        return

    baseline = arms[0]

    sep = "," if csv else "  "
    header = sep.join([
        f"{'arm':30s}", f"{'seeds':>5s}",
        f"{'pass@1':>8s}", f"{'sem':>6s}",
        f"{'pass@16':>8s}", f"{'sem':>6s}",
        f"{'peak@16':>8s}", f"{'@gen':>5s}",
        f"{'Δ@16':>8s}", f"{'Δ/sem':>6s}",
        f"{'wall_s':>7s}",
    ])
    print(header)
    if not csv:
        print("-" * len(header))

    def _peak_pass16(run):
        """Best pass@16 across all checkpoints for a single run."""
        best, best_gen = run["final"]["pass@16"], run["config"].get("generations", -1)
        for cp in run.get("checkpoints", []):
            if cp.get("pass@16", 0) > best:
                best = cp["pass@16"]
                best_gen = cp["gen"]
        return best, best_gen

    base_p16 = np.array([r["final"]["pass@16"] for r in results[baseline]])
    base_p16_mean = base_p16.mean()
    base_p16_sem = base_p16.std(ddof=1) / len(base_p16)**0.5 if len(base_p16) > 1 else 0

    for arm in arms:
        runs = results[arm]
        n = len(runs)
        p1 = np.array([r["final"]["pass@1"] for r in runs])
        p16 = np.array([r["final"]["pass@16"] for r in runs])
        peaks = [_peak_pass16(r) for r in runs]
        peak16 = np.array([p[0] for p in peaks])
        peak_gens = [p[1] for p in peaks]
        wall = np.array([r["wall_seconds"] for r in runs])

        p1_mean = p1.mean()
        p1_sem = p1.std(ddof=1) / n**0.5 if n > 1 else 0
        p16_mean = p16.mean()
        p16_sem = p16.std(ddof=1) / n**0.5 if n > 1 else 0

        delta = p16_mean - base_p16_mean
        # Pooled SEM for the difference
        pooled_sem = (p16_sem**2 + base_p16_sem**2)**0.5 if (p16_sem + base_p16_sem) > 0 else 1
        delta_sem = delta / pooled_sem if pooled_sem > 1e-9 else 0

        median_peak_gen = int(np.median(peak_gens))

        row = sep.join([
            f"{arm:30s}", f"{n:>5d}",
            f"{p1_mean:>8.4f}", f"{p1_sem:>6.3f}",
            f"{p16_mean:>8.4f}", f"{p16_sem:>6.3f}",
            f"{peak16.mean():>8.4f}", f"{median_peak_gen:>5d}",
            f"{delta:>+8.4f}", f"{delta_sem:>+6.1f}",
            f"{wall.mean():>7.1f}",
        ])
        print(row)


def print_trajectories(results):
    """Print pass@1 and pass@16 trajectories at checkpoint generations."""
    print("\n--- Trajectories (mean across seeds) ---\n")

    arm_order = ["iid_fixed_swept", "iid_resolution", "partitioned_resolution",
                 "partitioned_selective", "novelty_resolution", "orthogonal_resolution"]
    arms = [a for a in arm_order if a in results]

    # Collect all checkpoint generations
    all_gens = set()
    for arm in arms:
        for run in results[arm]:
            for cp in run["checkpoints"]:
                all_gens.add(cp["gen"])
    gens = sorted(all_gens)

    if not gens:
        return

    # pass@16 trajectory
    header = f"{'gen':>5s}" + "".join(f"  {a[:20]:>20s}" for a in arms)
    print("pass@16:")
    print(header)
    for g in gens:
        row = f"{g:>5d}"
        for arm in arms:
            vals = []
            for run in results[arm]:
                for cp in run["checkpoints"]:
                    if cp["gen"] == g and "pass@16" in cp:
                        vals.append(cp["pass@16"])
            if vals:
                row += f"  {np.mean(vals):>20.4f}"
            else:
                row += f"  {'—':>20s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CPU benchmark results")
    parser.add_argument("results_dir", help="Directory with JSON result files")
    parser.add_argument("--csv", action="store_true", help="CSV output")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No JSON files found in {args.results_dir}")
        sys.exit(1)

    total_runs = sum(len(v) for v in results.values())
    print(f"Loaded {total_runs} runs across {len(results)} arms\n")
    print_table(results, csv=args.csv)
    if not args.csv:
        print_trajectories(results)


if __name__ == "__main__":
    main()
