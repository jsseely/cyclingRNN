#!/vega/zmbbi/users/jss2219/miniconda2/envs/tensorflow/bin/python

import numpy as np
import pickle
import scipy.io as sio
from cycling_rnn import run_rnn

npsave_prefix = '/vega/zmbbi/users/jss2219/cyclingRNN/npsaves/'

monkey='D'
beta1 = np.logspace(-3,0,7)
beta2 = np.logspace(-3,0,7)
learning_rate = 0.0003
num_iters=5000
load_prev=False
load_model_path=None
local_machine=False

y_tf = np.zeros((beta1.size, beta2.size), dtype=object)
x_tf = np.zeros((beta1.size, beta2.size), dtype=object)
loss_val = np.zeros((beta1.size, beta2.size), dtype=object)

for i, i_val in enumerate(beta1):
  for j, j_val in enumerate(beta2):
    print i,i_val
    print j,j_val
    y_tf[i,j], x_tf[i,j], loss_val[i,j] = run_rnn(monkey=monkey,
                                                  beta1=i_val,
                                                  beta2=j_val,
                                                  learning_rate=learning_rate,
                                                  num_iters=num_iters,
                                                  load_prev=load_prev,
                                                  load_model_path=load_model_path,
                                                  local_machine=local_machine)
    np.save(npsave_prefix+'y', y_tf)
    np.save(npsave_prefix+'x', x_tf)

