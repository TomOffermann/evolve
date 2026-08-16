# CPU Benchmark

All ES arms on the countdown post-training task (94k-param TinyPolicy, sparse
binary verifiable reward). Designed for high seed counts on CPU — each run takes
~15 minutes and produces a JSON result file.

## Arms

| idx | name | sampler | weighting | sigma | status |
|---|---|---|---|---|---|
| 0 | iid_fixed_swept | IID | Rank | Fixed 0.005 | baseline |
| 1 | iid_resolution | IID | Rank | ResolutionRule | self-adapting baseline |
| 2 | partitioned_resolution | Partitioned(n=1) | Rank | ResolutionRule | E8-validated: +5.0 sem |
| 3 | partitioned_selective | Partitioned(n=1) | Rank | ResolutionRule | DiPEC-inspired: utility-gated |
| 4 | novelty_resolution | IID | Novelty(λ=0.2) | ResolutionRule | E7 reference |
| 5 | orthogonal_resolution | Orthogonal | Rank | ResolutionRule | untested |

## Run locally

```bash
# One arm, one seed
python code/cpu_benchmark/run.py --arm 2 --seed 0

# All arms, one seed
python code/cpu_benchmark/run.py --arm all --seed 0

# Aggregate results
python code/cpu_benchmark/aggregate.py code/cpu_benchmark/results/
```

## Run on Euler (ETH)

```bash
# First time: SSH to euler.ethz.ch, clone the repo, install deps
ssh euler.ethz.ch
module load python/3.11
git clone <repo-url> evolve && cd evolve
pip install --user torch numpy
mkdir -p logs

# Submit all 96 jobs (6 arms × 16 seeds)
sbatch code/cpu_benchmark/slurm/euler_array.sh

# Monitor
squeue -u $USER

# Collect results
python code/cpu_benchmark/aggregate.py code/cpu_benchmark/results/
```

## Output format

Each run produces `results/<arm>_seed<N>.json`:

```json
{
  "arm": "partitioned_resolution",
  "seed": 3,
  "config": {"n_pop": 128, "rank": 1, "sigma_init": 0.005, ...},
  "checkpoints": [
    {"gen": 0, "pass@1": 0.043, "pass@4": 0.112, "pass@16": 0.211},
    {"gen": 50, "pass@1": 0.067, "pass@4": 0.145, "pass@16": 0.224},
    ...
  ],
  "final": {"pass@1": 0.112, "pass@16": 0.232},
  "history": [{"gen": 0, "fitness_mean": ..., "sigma": ...}, ...],
  "wall_seconds": 342.5
}
```
