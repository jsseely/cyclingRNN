"""
  after a RUN is complete, run this script to perform analyses. saves a pandas dataframe with results.

  usage:
  python analyze_run.py

"""

import numpy as np
import scipy.io as sio

from sklearn import metrics
import pandas as pd

import pickle
import os

import cyclingrnn.geometric as geo
from cyclingrnn import sigerr
from cyclingrnn.train import *

def get_full_data(cur_params):
  """
    returns u_data, y_data for tensorflow's feed_dict.
  """
  path_prefix = './'
  if cur_params['monkey']=='D':
    data = sio.loadmat(path_prefix+'drakeFeb.mat')
  else:
    data = sio.loadmat(path_prefix+'cousFeb.mat')

  # Built augmented data
  emg2 = preprocess_array(data['D'][0, 0]['EMG'])
  time_axis, time_inds1, time_inds2 = get_time_axis(data['D'][0, 0]['KIN'])
  y_data1 = emg2[time_axis]
  p = y_data1.shape[-1]

  # Build inputs
  m = 2
  u_data1 = create_input_array(y_data1.shape)

  # Augmented data
  # For regularizing the network -- it must fit actual and augmented data
  period = int(np.round(np.diff(time_inds2).mean()))
  y_cat1 = augmented_data(emg2, time_inds1, period=period, tiles=10)
  y_cat1 = y_cat1[::25]
  y_cat2 = augmented_data(emg2, time_inds2, period=period, tiles=10)
  y_cat2 = y_cat2[::25]

  u_cat1 = create_input_array(y_cat1.shape)
  u_cat2 = create_input_array(y_cat2.shape)

  sequence_length = [y_data1.shape[0], y_cat1.shape[0], y_cat2.shape[0]]
  y_data = np.zeros((np.max(sequence_length), 4*3, p))
  u_data = np.zeros((np.max(sequence_length), 4*3, m))

  y_data[:sequence_length[0], 0:4, :] = y_data1
  y_data[:sequence_length[1], 4:8, :] = y_cat1
  y_data[:sequence_length[2], 8:12, :] = y_cat2

  u_data[:sequence_length[0], 0:4, :] = u_data1
  u_data[:sequence_length[1], 4:8, :] = u_cat1
  u_data[:sequence_length[2], 8:12, :] = u_cat2
  return u_data, y_data

def mean_squared_error(x, y):
  return np.mean((x - y)**2)

def get_noise_sim(stddev_struct, stddev_state, RUN, sim, y_data, u_data):
  """
    loads TF model. Then simulates it again, but with noise
    stddev_struct: std dev of structural perturbation on weight matrix
    stddev_state: std dev of additive noise to each neuron
    RUN: directory of the current run
    sim: integer for the simulation
    y_data, u_data: reference data for feed_dict
  """
  TF_PATH = RUN+'tfsaves/'+str(sim)
  tf.reset_default_graph()
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(TF_PATH+'.meta')
    new_saver.restore(sess, TF_PATH)
    # Get Mat variable
    Mat = sess.run([v for v in tf.global_variables() if v.name == 'RNN/BasicRNNCellNoise/Linear/Matrix:0'][0])
    A = Mat[2:]
    B = Mat[:2]
    newMat = np.zeros(Mat.shape)
    newMat[:2] = B
    newMat[2:] = A + stddev_struct*np.random.randn(A.shape[0], A.shape[1])
    sess.run([v for v in tf.global_variables() if v.name == 'RNN/BasicRNNCellNoise/Linear/Matrix:0'][0].assign(newMat))
    feed_dict = {'Y:0': y_data, 'U:0': u_data, 'stddev_state:0': stddev_state}
    y_hat, x_hat, loss = sess.run(['Y_hat:0', 'RNN/TensorArrayPack_1/TensorArrayGatherV2:0', 'cost1:0'], feed_dict)
  return y_hat, x_hat, loss

def get_noise_robustness(sim, y_data, u_data, conds, emg, err_f=sigerr.mse_sigerr, dtw_err=False, num=10, trials=4):
  """
    gradually increases noise until error (R squared) drops below 0.5
    tests for both state noise robustness as well as structural robustness
    - state noise robustness: noise of strength sdtdev_state injected into neurons during simulation
    - structural robustness: apply a single noisy perturbation to the matrix A then simulate
    sim: int. which simulation to use.
    y_data, u_data: reference data for feed_dict
    num: input for np.logspace(), number of stddevs...
    
  """
  
  TF_PATH = RUN+'tfsaves/'+str(sim)
  tf.reset_default_graph()
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(TF_PATH+'.meta')
    new_saver.restore(sess, TF_PATH)
    # Get Mat variable
    Mat = sess.run([v for v in tf.global_variables() if v.name == 'RNN/BasicRNNCellNoise/Linear/Matrix:0'][0]) # can just run with the string... 
    A = Mat[2:]
    B = Mat[:2]

    errors_state = np.zeros((num, trials))
    errors_struct = np.zeros((num, trials))
    
    stddev_struct = 0.
    stddev_state = 0.
    for i_state, stddev_state in enumerate(np.logspace(-3, 2, num=num)):
      for tr in range(trials):
        newMat = np.zeros(Mat.shape)
        newMat[:2] = B
        newMat[2:] = A + stddev_struct*np.random.randn(A.shape[0], A.shape[1])
        sess.run([v for v in tf.global_variables() if v.name == 'RNN/BasicRNNCellNoise/Linear/Matrix:0'][0].assign(newMat))
        feed_dict = {'Y:0': y_data, 'U:0': u_data, 'stddev_state:0': stddev_state}
        y_hat = sess.run('Y_hat:0', feed_dict)
        if dtw_err:
          dist, err = err_f(y_hat[:emg.shape[0], conds, :], emg[:, conds, :])
        else:
          err = err_f(y_hat[:emg.shape[0], conds, :], emg[:, conds, :])
        errors_state[i_state, tr] = err
      if np.mean(errors_state[i_state, :]) < 0.5:
        break

    stddev_state_out = stddev_state
    stddev_struct = 0.
    stddev_state = 0.
    for i_struct, stddev_struct in enumerate(np.logspace(-5, 0, num=num)):
      for tr in range(trials):
        newMat = np.zeros(Mat.shape)
        newMat[:2] = B
        newMat[2:] = A + stddev_struct*np.random.randn(A.shape[0], A.shape[1])
        sess.run([v for v in tf.global_variables() if v.name == 'RNN/BasicRNNCellNoise/Linear/Matrix:0'][0].assign(newMat))
        feed_dict = {'Y:0': y_data, 'U:0': u_data, 'stddev_state:0': stddev_state}
        y_hat = sess.run('Y_hat:0', feed_dict)
        if dtw_err:
          dist, err = err_f(y_hat[:emg.shape[0], conds, :], emg[:, conds, :])
        else:
          err = err_f(y_hat[:emg.shape[0], conds, :], emg[:, conds, :])
        errors_struct[i_struct, tr] = err
      if np.mean(errors_struct[i_struct, :]) < 0.5:
        break
  
  return stddev_state_out, stddev_struct

def evaluate_run(conds, RUN):
  """
    Evaluate a RUN (all simulations) for a specified choice of conditions (conds)
  """
  # process directory information
  data_files = [i for i in os.listdir(RUN+'npsaves/') if i.endswith('x.npy')]
  sim_nums = [i.replace('x.npy', '') for i in data_files]
  param_files = [i+'params.pickle' for i in sim_nums]

  assert len(data_files) == len(param_files)

  total_sims = len(data_files)

  # Get monkey
  cur_params = pickle.load(open(RUN+'npsaves/'+param_files[0], 'rb'))
  # build input and output data
  if cur_params['monkey']=='D':
    data = sio.loadmat('./drakeFeb_processed.mat')
  else:
    data = sio.loadmat('./cousFeb_processed.mat')
  emg = data['EMG']
  m1 = data['M1']
  u_full, emg_full = get_full_data(cur_params)

  params = []
  out_metrics = []
  for iteration, sim in enumerate([int(i) for i in sim_nums]):
    cur_params = pickle.load(open(RUN+'npsaves/'+str(sim)+'params.pickle', 'rb'))
    x = np.load(RUN+'npsaves/'+str(sim)+'x.npy')
    y = np.load(RUN+'npsaves/'+str(sim)+'y.npy')
    # truncate to remove augmented data
    x_trunc = x[:emg.shape[0], :emg.shape[1], :]
    y_trunc = y[:emg.shape[0], :emg.shape[1], :]
    u_trunc = create_input_array(y_trunc.shape)

    # truncate conditions
    x_trunc = x_trunc[:, conds, :]
    y_trunc = y_trunc[:, conds, :]
    u_trunc = u_trunc[:, conds, :]

    emg_ = emg[:, conds, :]

    # get params
    params.append(cur_params)

    try:
      mets = dict()
      R2 = metrics.r2_score(np.reshape(emg_, [-1, emg_.shape[-1]]),
                            np.reshape(y_trunc, [-1, y_trunc.shape[-1]]), multioutput='uniform_average') # condition truncation on emg
      if R2 > 0.5:
        mets['sim_num'] = sim

        mets['percent_tangling1_01']  = geo.percent_tangling( x_trunc, emg_, th=1, alpha=0.1 ) # note condition truncation on emg
        mets['percent_tangling2_01']  = geo.percent_tangling( x_trunc, emg_, th=2, alpha=0.1 ) # note condition truncation on emg
        mets['percent_tangling3_01']  = geo.percent_tangling( x_trunc, emg_, th=3, alpha=0.1 ) # note condition truncation on emg
        mets['percent_tangling1_001'] = geo.percent_tangling( x_trunc, emg_, th=1, alpha=0.01 ) # note condition truncation on emg
        mets['percent_tangling2_001'] = geo.percent_tangling( x_trunc, emg_, th=2, alpha=0.01 ) # note condition truncation on emg
        mets['percent_tangling3_001'] = geo.percent_tangling( x_trunc, emg_, th=3, alpha=0.01 ) # note condition truncation on emg

        mets['tangling_90_01']  = geo.tangling_cdf( x_trunc, cutoff=0.90, alpha=0.1  )
        mets['tangling_90_001'] = geo.tangling_cdf( x_trunc, cutoff=0.95, alpha=0.01 )
        mets['tangling_95_01']  = geo.tangling_cdf( x_trunc, cutoff=0.90, alpha=0.1  )
        mets['tangling_90_001'] = geo.tangling_cdf( x_trunc, cutoff=0.95, alpha=0.01 )
        
        mets['path_length'] = np.sum(geo.get_path_length(x_trunc, filt_freq=0.25))
        mets['mean_curvature'] = geo.mean_curvature(x_trunc, total_points=11, deg=4, normalize=True)

        mets['MSE'] = metrics.mean_squared_error(np.reshape(emg_, [-1,emg_.shape[-1]]),
                                                  np.reshape(y_trunc, [-1, y_trunc.shape[-1]]), multioutput='uniform_average') # condition truncation on emg
        mets['R2'] = R2
        
        err_fun = sigerr.dtw_sigerr_P
        mets['noise_robustness'], mets['struct_robustness'] = get_noise_robustness(sim, emg_full, u_full,
                                                                                   conds, emg, err_fun, dtw_err=True, num=100, trials=5)
        out_metrics.append(mets)
    except:
      pass

    if iteration % 50 == 0:
      print 'Sim completed: ', sim
    
  return params, out_metrics

def process_dataframe(params, out_metrics):
  """
    panda panda panda
  """
  # join params and metrics into one dataframe
  df = pd.DataFrame(params)
  metrics_df = pd.DataFrame(out_metrics)
  df = df.join(metrics_df)
  
  # remove bad fits
  #inds = df['R2'] > 0.5
  #df = df[inds]
  
  #Split the generalized curvature columns into multiple columns, one for each curvature
  mc = df['mean_curvature'].apply(pd.Series)
  mc.rename(columns={0: 'mean_curvature'}, inplace=True)
  mc.rename(columns={1: 'mean_torsion'}, inplace=True)
  mc = mc[['mean_curvature', 'mean_torsion']]
  df = df.drop('mean_curvature', 1)
  df = df.join(mc)
  
  return df


RUNset = ['./saves/170218D/', './saves/170218C/']
condset = [[0,1], [0,1,2,3]]

for RUN in RUNset:
  for conds in condset:
    print 'RUN: ', RUN
    print 'conds: ', conds
    params, out_metrics = evaluate_run(conds, RUN)
    df = process_dataframe(params, out_metrics)
    pickle.dump(df, open(RUN+'df'+''.join(map(str,conds))+'.pickle', 'wb'))
