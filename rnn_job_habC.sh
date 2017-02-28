#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=cyclingRNN_C
#SBATCH -c 1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=3gb
#SBATCH --array=1-300
#SBATCH --error=./slurmerr/%A_%a.err
#SBATCH --output=./slurmout/%A_%a.out

python wrapper.py $SLURM_ARRAY_TASK_ID "170222C" "C"