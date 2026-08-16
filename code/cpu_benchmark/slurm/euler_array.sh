#!/bin/bash
#SBATCH --job-name=evolve-cpu
#SBATCH --array=0-95               # 6 arms × 16 seeds
#SBATCH --time=01:00:00            # ~15 min per run, generous margin
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evolve_%A_%a.out
#SBATCH --error=logs/evolve_%A_%a.err
#
# CPU benchmark on Euler — one (arm, seed) per array task.
#
# Setup (run once):
#   ssh euler.ethz.ch              # first login creates your account
#   module load python/3.11        # CPU-only Python with PyTorch
#   git clone <repo-url> evolve
#   cd evolve
#   pip install --user torch numpy
#   mkdir -p logs
#
# Submit:
#   sbatch code/cpu_benchmark/slurm/euler_array.sh
#
# Collect results:
#   python code/cpu_benchmark/aggregate.py code/cpu_benchmark/results/

N_ARMS=6
N_SEEDS=16

ARM_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))

echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${ARM_IDX} seed=${SEED}"
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

module load python/3.11 2>/dev/null || true

python code/cpu_benchmark/run.py \
    --arm "$ARM_IDX" \
    --seed "$SEED" \
    --output code/cpu_benchmark/results/
