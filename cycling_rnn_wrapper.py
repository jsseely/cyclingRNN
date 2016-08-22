"""
Docstring
"""
import datetime
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import itertools
import pickle
import copy

from cycling_rnn import run_rnn
from sklearn.grid_search import ParameterGrid


# Iterable Parameters
# Indexing will be ordered alphabetically
# param_grid = {}
# param_grid['activation'] = [lambda x: x, tf.tanh]
# param_grid['beta1'] = [0, 3, 10, 30, 100, 300, 1000]
# param_grid['beta2'] = [0, 3, 10, 30, 100, 300, 1000]
# param_grid['monkey'] =['C', 'D']
# param_grid['num_neurons'] = [10, 100, 1000]
# 
# val_lengths = [len(v) for k,v in sorted(param_grid.items())]

param_grid = {}
param_grid['activation'] = [tf.tanh]
param_grid['beta1'] = [0]
param_grid['beta2'] = [0]
param_grid['monkey'] =['D', 'C']
param_grid['num_neurons'] = [100]

val_lengths = [len(v) for k,v in sorted(param_grid.items())]

# Make a copy of param_grid to export to matlab.
# Dicts are unordered, so we need to ensure matlab gets the right order.
param_grid_matlab = copy.deepcopy(param_grid)
param_grid_matlab['val_lengths'] = val_lengths
param_grid_matlab['sorted'] = sorted(param_grid)

# val_lengths ordered by keys
# Important: apparently ParameterGrid() orders lexicographically based on keys.
# This should be triple checked.

# Fixed parameters
LEARNING_RATE = 0.0003
NUM_ITERS = 1000
LOAD_PREV = False
PREV_PATH = None
LOCAL_MACHINE = True

# Current run and paths
CUR_RUN = str(datetime.datetime.now().strftime("%m%d-%H%M-%S"))

if LOCAL_MACHINE:
  PATH_PREFIX = '/Users/jeff/Documents/Python/_projects/cyclingRNN/'
  TB_PREFIX = '/tmp/tf/'
  ML_PREFIX = '/Users/jeff/Documents/MATLAB/cyclingRNN/data/tf_'
else:
  PATH_PREFIX = '/vega/zmbbi/users/jss2219/cyclingRNN/'
  TB_PREFIX = PATH_PREFIX+'tensorboard/'
  ML_PREFIX = PATH_PREFIX+'matlab/tf_'

NPSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/npsaves/'
TFSAVE_PREFIX = PATH_PREFIX+'saves/'+CUR_RUN+'/tfsaves/'
TB_PREFIX = TB_PREFIX+CUR_RUN+'/'
ML_PREFIX = ML_PREFIX+CUR_RUN+'/'

os.makedirs(NPSAVE_PATH)
os.makedirs(TFSAVE_PREFIX)
os.makedirs(ML_PREFIX)

# save paraM-grid
pickle.dump(param_grid, open(NPSAVE_PATH+'param_grid.pickle', 'wb'))

#TODO: write parameters to output text file in ./CUR_RUN/

Y_TF = np.zeros(val_lengths, dtype=object)
X_TF = np.zeros(val_lengths, dtype=object)

print 'Current Run: '+CUR_RUN

for i, cur_params in enumerate(ParameterGrid(param_grid)):
  print cur_params
  param_inds = np.unravel_index(i, val_lengths) # TODO: 'C' or 'F'? Pretty sure this is right.
  tb_path = TB_PREFIX+str(i)
  tfsave_path = TFSAVE_PREFIX+str(i)
  if LOAD_PREV:
    load_model_path = PATH_PREFIX+'saves/'+PREV_PATH+'/tfsaves/'+str(i)
  else:
    load_model_path = None

  Y_TF[param_inds], X_TF[param_inds] = run_rnn(monkey=cur_params['monkey'],
                                               beta1=cur_params['beta1'],
                                               beta2=cur_params['beta2'],
                                               activation=cur_params['activation'],
                                               num_neurons=cur_params['num_neurons'],
                                               learning_rate=LEARNING_RATE,
                                               num_iters=NUM_ITERS,
                                               load_prev=LOAD_PREV,
                                               save_model_path=tfsave_path,
                                               load_model_path=load_model_path,
                                               tb_path=tb_path,
                                               local_machine=LOCAL_MACHINE)

  np.save(NPSAVE_PATH+'y', Y_TF)
  np.save(NPSAVE_PATH+'x', X_TF)
  sio.savemat(ML_PREFIX+'data.mat', mdict={'X': X_TF, 'Y': Y_TF, 'params': param_grid_matlab})

