#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=cyclingRNN_C
#SBATCH -c 1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000mb
#SBATCH --array=1-5

python wrapper.py $SLURM_ARRAY_TASK_ID "TestA" "C"