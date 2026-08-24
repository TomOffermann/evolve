#!/bin/bash
#SBATCH --job-name=evolve-new
#SBATCH --array=0-15               # 4 new arms × 4 seeds (smoke test)
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/evolve_%A_%a.out
#SBATCH --error=logs/evolve_%A_%a.err
#
# Smoke test for new arms only (6-9): guided, archive, fine_p8, fine_p16.
# 4 seeds each. Takes ~15 min.
#
# Submit:   sbatch code/cpu_benchmark/slurm/euler_new_arms.sh
# Results:  python code/cpu_benchmark/aggregate.py code/cpu_benchmark/results/

N_ARMS=4     # arms 6, 7, 8, 9
N_SEEDS=4

ARM_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS + 6))   # offset to start at arm 6
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))

echo "Task ${SLURM_ARRAY_TASK_ID}: arm=${ARM_IDX} seed=${SEED}"
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

module load stack/2024-06 gcc/12.2.0 python/3.11.6 2>/dev/null || true

python code/cpu_benchmark/run.py \
    --arm "$ARM_IDX" \
    --seed "$SEED" \
    --output code/cpu_benchmark/results/
