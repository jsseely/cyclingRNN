"""
  A wrapper function to specify hyperparameters and loop through run_rnn
"""
import datetime
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import pickle
import copy

from cycling_rnn import run_rnn
from sklearn.grid_search import ParameterGrid

# Iterable Parameters
# Identity activation is: [lambda x: x]
param_grid = {}
param_grid['activation'] = [tf.tanh]
param_grid['beta1'] = [0]
param_grid['beta2'] = [0]
param_grid['stddev_state'] = [0]
param_grid['stddev_out'] = [0]
param_grid['monkey'] =['D']
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
NUM_ITERS = 10000
LOAD_PREV = False
PREV_PATH = None
LOCAL_MACHINE = False

# Current run and paths
CUR_RUN = str(datetime.datetime.now().strftime("%m%d-%H%M-%S"))

if LOCAL_MACHINE:
  PATH_PREFIX = '/Users/jeff/Documents/Python/_projects/cyclingRNN/'
else:
  PATH_PREFIX = '/vega/zmbbi/users/jss2219/cyclingRNN/'

NPSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/npsaves/'
TFSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/tfsaves/'
MLSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/mlsaves/'
TB_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/tb/'

os.makedirs(NPSAVE_PATH)
os.makedirs(TFSAVE_PATH)
os.makedirs(ML_PREFIX)

# save param-grid
pickle.dump(param_grid, open(NPSAVE_PATH+'param_grid.pickle', 'wb'))

#TODO: write parameters to output text file in ./CUR_RUN/

print 'Current Run: '+CUR_RUN

for i, cur_params in enumerate(ParameterGrid(param_grid)):
  # Note: param_inds = np.unravel_index(i, val_lengths)
  print cur_params
  if LOAD_PREV:
    load_model_path = PATH_PREFIX+'saves/'+PREV_PATH+'/tfsaves/'+str(i)
  else:
    load_model_path = None

  #TODO: just have run_rnn take in a param dict.
  Y_TF, X_TF = run_rnn(monkey=cur_params['monkey'],
                       beta1=cur_params['beta1'],
                       beta2=cur_params['beta2'],
                       stddev_state=cur_params['stddev_state'],
                       stddev_out=cur_params['stddev_out'],
                       activation=cur_params['activation'],
                       num_neurons=cur_params['num_neurons'],
                       learning_rate=LEARNING_RATE,
                       num_iters=NUM_ITERS,
                       load_prev=LOAD_PREV,
                       save_model_path=TFSAVE_PATH+str(i),
                       load_model_path=load_model_path,
                       tb_path=TB_PATH+str(i),
                       local_machine=LOCAL_MACHINE)

  # TODO: just save individual arrays, not a cell array...
  # Later, combine together if needed.
  np.save(NPSAVE_PATH+str(i)+'y', Y_TF)
  np.save(NPSAVE_PATH+str(i)+'x', X_TF)
  sio.savemat(MLSAVE_PATH+str(i)+'.mat', mdict={'X': X_TF, 'Y': Y_TF, 'params': param_grid_matlab})
