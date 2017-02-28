"""
  A wrapper function to specify hyperparameters and loop through run_rnn
  Customized for the Yeti HPC to take in array submissions.
  See rnn_job.sh

  usage:
  python wrapper.py 1 "20170101" "C"

  Args:
    i, int
    CUR_RUN, string
    monkey, 'C' or 'D'

"""
import sys
import os
import numpy as np
import scipy.io as sio
import pickle
import random
from cyclingrnn.train import train_rnn

# hyperparameters
cur_params = {}
cur_params['monkey']        = str(sys.argv[3])
cur_params['beta1']         = 10**np.random.uniform(-3, 3)
cur_params['beta2']         = 10**np.random.uniform(-3, 3)
cur_params['activation']    = random.choice(['tanh'])
cur_params['stddev_state']  = 10**np.random.uniform(-6, -3)
cur_params['stddev_out']    = 0.
cur_params['num_neurons']   = 100
cur_params['rnn_init']      = random.choice(['orth'])
cur_params['learning_rate'] = 10**np.random.uniform(-5, -3)

# fixed parameters
#LEARNING_RATE = 0.0003
NUM_ITERS = 50000
LOAD_PREV = False

# current run and paths
CUR_SIM = int(sys.argv[1])
CUR_RUN = str(sys.argv[2])

print CUR_SIM

NPSAVE_PATH = './saves/'+CUR_RUN+'/npsaves/'
TFSAVE_PATH = './saves/'+CUR_RUN+'/tfsaves/'
MLSAVE_PATH = './saves/'+CUR_RUN+'/mlsaves/'
TB_PATH = './saves/'+CUR_RUN+'/tb/'

def make_dir(path):
  """like os.makedirs(path) but avoids race conditions"""
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise

make_dir(NPSAVE_PATH)
make_dir(TFSAVE_PATH)
make_dir(MLSAVE_PATH)
make_dir(TB_PATH)

print 'Current Run: '+CUR_RUN

# save params
pickle.dump(cur_params, open(NPSAVE_PATH+str(CUR_SIM)+'params.pickle', 'wb'))

print cur_params

Y_TF, X_TF = train_rnn(monkey=cur_params['monkey'],
                       beta1=cur_params['beta1'],
                       beta2=cur_params['beta2'],
                       stddev_state=cur_params['stddev_state'],
                       stddev_out=cur_params['stddev_out'],
                       activation=cur_params['activation'],
                       rnn_init=cur_params['rnn_init'],
                       num_neurons=cur_params['num_neurons'],
                       learning_rate=cur_params['learning_rate'],
                       num_iters=NUM_ITERS,
                       save_model_path=TFSAVE_PATH+str(CUR_SIM),
                       tb_path=TB_PATH+str(CUR_SIM))

np.save(NPSAVE_PATH+str(CUR_SIM)+'y', Y_TF)
np.save(NPSAVE_PATH+str(CUR_SIM)+'x', X_TF)
sio.savemat(MLSAVE_PATH+str(CUR_SIM)+'.mat', mdict={'X': X_TF, 'Y': Y_TF, 'params': cur_params})
