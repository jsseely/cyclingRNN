#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=a_run_rnn
#SBATCH --cpus-per-task 5
#SBATCH --nodes 10
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4gb
#SBATCH --error=./slurmerr/%A_%a.err
#SBATCH --output=./slurmout/%A_%a.out

python analyze_run.py
