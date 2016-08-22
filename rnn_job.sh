#!/bin/bash

#PBS -N cyclingRNN
#PBS -W group_list=yetizmbbi
#PBS -l walltime=10:00:00,mem=2000mb
#PBS -M jss2219@columbia.edu
#PBS -m abe
#PBS -V

#PBS -o localhost:/vega/zmbbi/users/jss2219/cyclingRNN/output/
#PBS -e localhost:/vega/zmbbi/users/jss2219/cyclingRNN/error/

source activate tensorflow

LD_LIBRARY_PATH="/vega/zmbbi/users/jss2219/miniconda2/envs/tensorflow/bin/lib/x86_64-linux-gnu/:/vega/zmbbi/users/jss2219/miniconda2/envs/tensorflow/bin/usr/lib64/" /vega/zmbbi/users/jss2219/miniconda2/envs/tensorflow/bin/lib/x86_64-linux-gnu/ld-2.17.so `which python` cycling_rnn_wrapper.py