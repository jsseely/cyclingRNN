"""
  Random hyperparameters (12-14-16 version of cycling_rnn_wrapper_yeti). Uses random hyperparameters instead of a range.
  This one is for 'C'

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

param_grid['activation']    = np.random.choice(['tanh'], size=1)
param_grid['beta1']         = 10**np.random.uniform(-6, 2, size=1)
param_grid['beta2']         = 10**np.random.uniform(-6, 2, size=1)
param_grid['stddev_state']  = 10**np.random.uniform(-6, 2, size=1)
#param_grid['stddev_out']    = 10**np.random.uniform(-4, 0, size=1)
param_grid['stddev_out']    = np.random.choice([0.], size=1)
param_grid['monkey']        = np.random.choice(['C'], size=1)
param_grid['num_neurons']   = np.array([100]).astype(int)

cur_params = ParameterGrid(param_grid)[0]

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

print 'Current Run: '+CUR_RUN

i = int(sys.argv[1])

# save params
pickle.dump(cur_params, open(NPSAVE_PATH+str(i)+'params.pickle', 'wb'))

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
sio.savemat(MLSAVE_PATH+str(i)+'.mat', mdict={'X': X_TF, 'Y': Y_TF, 'params': cur_params})

