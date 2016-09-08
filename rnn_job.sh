#!/bin/bash
#PBS -N cyclingRNN
#PBS -W group_list=yetizmbbi
#PBS -l walltime=24:00:00,mem=2000mb
#PBS -M jss2219@cumc.columbia.edu
#PBS -m n
#PBS -V
#PBS -t 0-106
#PBS -o /vega/zmbbi/users/jss2219/cyclingRNN/output/
#PBS -e /vega/zmbbi/users/jss2219/cyclingRNN/error/

# Reminder: use source activate tensorflow before running:
# source /vega/zmbbi/users/jss2219/miniconda2/bin/activate tensorflow

LD_LIBRARY_PATH="/vega/zmbbi/users/jss2219/glibc/lib/x86_64-linux-gnu/:/vega/zmbbi/users/jss2219/glibc/usr/lib64/" /vega/zmbbi/users/jss2219/glibc/lib/x86_64-linux-gnu/ld-2.17.so `which python` cycling_rnn_wrapper_yeti.py $PBS_ARRAYID "8-29"
