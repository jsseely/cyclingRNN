"""
  A wrapper function to specify hyperparameters and loop through run_rnn
  Customized for the Yeti HPC to take in array submissions.
  See rnn_job.sh
"""
import datetime
import os
import numpy as np
import scipy.io as sio
import pickle
import copy

from cycling_rnn import run_rnn, parameter_grid_split
from sklearn.grid_search import ParameterGrid

import sys

# TODO: find better way to get len(ParameterGrid(param_grid)) for bash script $ PBS -t 1-x...

# Iterable Parameters
param_grid = {}
param_grid['activation'] = ['tanh', 'linear']
param_grid['beta1'] = np.concatenate((np.array([0]), np.logspace(-7, -2, 21)), axis=0)
param_grid['beta2'] = np.concatenate((np.array([0]), np.logspace(-5, 0, 21)), axis=0)
param_grid['stddev_state'] = np.concatenate((np.array([0]), np.logspace(-5, 0, 21)), axis=0)
param_grid['stddev_out'] = np.concatenate((np.array([0]), np.logspace(-4, 1, 21)), axis=0)
param_grid['monkey'] =['D']
param_grid['num_neurons'] = np.concatenate((np.array([100]).astype(int), np.round(np.logspace(1, 3, 21)).astype(int)), axis=0)

val_lengths = [len(v) for k, v in sorted(param_grid.items())]

# Make a copy of param_grid to export to matlab.
param_grid_matlab = copy.deepcopy(param_grid)
param_grid_matlab['val_lengths'] = val_lengths
param_grid_matlab['sorted'] = sorted(param_grid)

# val_lengths ordered by keys
# Important: apparently ParameterGrid() orders lexicographically based on keys.

# Split param_grid to search along hyperparameter axes only
param_grid = parameter_grid_split(param_grid)

# Fixed parameters
LEARNING_RATE = 0.0003
NUM_ITERS = 10000
LOAD_PREV = False
PREV_PATH = None
LOCAL_MACHINE = False

# Current run and paths
CUR_RUN = str(sys.argv[2])

if LOCAL_MACHINE:
  PATH_PREFIX = '/Users/jeff/Documents/Python/_projects/cyclingRNN/'
else:
  PATH_PREFIX = '/vega/zmbbi/users/jss2219/cyclingRNN/'

NPSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/npsaves/'
TFSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/tfsaves/'
MLSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/mlsaves/'
TB_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/tb/'

def make_dir(path):
  """
    like os.makedirs(path) but avoids race conditions
  """
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

make_dir(NPSAVE_PATH)
make_dir(TFSAVE_PATH)
make_dir(MLSAVE_PATH)

# save param-grid (split version)
pickle.dump(param_grid, open(NPSAVE_PATH+'param_grid.pickle', 'wb'))

print 'Current Run: '+CUR_RUN

i = int(sys.argv[1])
cur_params = ParameterGrid(param_grid)[i]

print cur_params

if LOAD_PREV:
  load_model_path = PATH_PREFIX+'saves/'+PREV_PATH+'/tfsaves/'+str(i)
else:
  load_model_path = None

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

np.save(NPSAVE_PATH+str(i)+'y', Y_TF)
np.save(NPSAVE_PATH+str(i)+'x', X_TF)
sio.savemat(MLSAVE_PATH+str(i)+'.mat', mdict={'X': X_TF, 'Y': Y_TF, 'params': param_grid_matlab})
