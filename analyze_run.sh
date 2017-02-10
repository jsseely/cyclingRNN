#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=cyclingRNN_C
#SBATCH -c 1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4gb
#SBATCH --error=./slurmerr/%A_%a.err
#SBATCH --output=./slurmout/%A_%a.out

python analyze_run.py