#!/bin/bash
#SBATCH --job-name=evolve-test
#SBATCH --array=0-7                # 2 arms × 4 seeds (smoke test)
#SBATCH --time=00:30:00            # 30 min, generous for a test
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evolve_%A_%a.out
#SBATCH --error=logs/evolve_%A_%a.err
#
# Smoke test: run 2 arms (iid_resolution + partitioned_resolution) × 4 seeds.
# Takes ~15 min. If this works, run the full euler_array.sh (96 jobs).
#
# Submit:   sbatch code/cpu_benchmark/slurm/euler_test.sh
# Monitor:  squeue -u $USER
# Results:  python code/cpu_benchmark/aggregate.py code/cpu_benchmark/results/

N_ARMS=2     # only arms 1 and 2
N_SEEDS=4

ARM_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS + 1))   # arms 1 and 2
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))

echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${ARM_IDX} seed=${SEED}"
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

module load python/3.11 2>/dev/null || true

python code/cpu_benchmark/run.py \
    --arm "$ARM_IDX" \
    --seed "$SEED" \
    --output code/cpu_benchmark/results/
