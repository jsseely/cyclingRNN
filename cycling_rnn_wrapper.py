import datetime
import os
import numpy as np
from cycling_rnn import run_rnn

# Parameters
MONKEY = 'D'
BETA1 = np.array([0])
BETA2 = np.array([0])

LEARNING_RATE = 0.0003
NUM_ITERS = 10000
LOAD_PREV = False
PREV_PATH = None
LOCAL_MACHINE = True

CUR_RUN = MONKEY+'_'+str(datetime.datetime.now().strftime("%m%d-%H%M-%S"))

if LOCAL_MACHINE:
  PATH_PREFIX = '/Users/jeff/Documents/Python/_projects/cyclingRNN/'
  TB_PREFIX = '/tmp/tf/'
else:
  PATH_PREFIX = '/vega/zmbbi/users/jss2219/cyclingRNN/'
  TB_PREFIX = PATH_PREFIX+'tensorboard/'

NPSAVE_PATH = PATH_PREFIX+'saves/'+CUR_RUN+'/npsaves/'
TFSAVE_PREFIX = PATH_PREFIX+'saves/'+CUR_RUN+'/tfsaves/'
TB_PREFIX = TB_PREFIX+CUR_RUN+'/'

os.makedirs(NPSAVE_PATH)
os.makedirs(TFSAVE_PREFIX)

#TODO: write parameters to output text file in ./CUR_RUN/

Y_TF = np.zeros((BETA1.size, BETA2.size), dtype=object)
X_TF = np.zeros((BETA1.size, BETA2.size), dtype=object)
LOSS_VAL = np.zeros((BETA1.size, BETA2.size), dtype=object)

for i, i_val in enumerate(BETA1):
  for j, j_val in enumerate(BETA2):
    print 'beta 1: %02d, %05f' % (i, i_val) # uh...
    print 'beta 2: %02d, %05f' % (j, j_val)
    hp_pf = '%02d_%02d' % (i, j)
    tb_path = TB_PREFIX+hp_pf # TODO: check -- with or without trailing '/'?
    tfsave_path = TFSAVE_PREFIX+hp_pf
    if LOAD_PREV:
      load_model_path = PATH_PREFIX+'saves/'+PREV_PATH+'/tfsaves/'+hp_pf
    else:
      load_model_path = None

    Y_TF[i, j], X_TF[i, j], LOSS_VAL[i, j] = run_rnn(monkey=MONKEY,
                                                     beta1=i_val,
                                                     beta2=j_val,
                                                     learning_rate=LEARNING_RATE,
                                                     num_iters=NUM_ITERS,
                                                     load_prev=LOAD_PREV,
                                                     save_model_path=tfsave_path,
                                                     load_model_path=load_model_path,
                                                     tb_path=tb_path,
                                                     local_machine=LOCAL_MACHINE)
    np.save(NPSAVE_PATH+'/y', Y_TF)
    np.save(NPSAVE_PATH+'/x', X_TF)