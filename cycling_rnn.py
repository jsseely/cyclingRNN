"""
  A collection of functions for the 'cycling RNN' project.
  run_rnn is the main function that builds the tensorflow graph and does the training.
"""
import datetime
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy import signal
from custom_rnn_cells import BasicRNNCellNoise
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy.polynomial.polynomial as P

import copy

import pdb

# TODO: write custom logspace function
# 0.001, .003, .01, .03, etc.
# rounding
# define granularity -- 0.01, 0.1 vs 0.01, 0.03, 0.1, etc

### PLOTTING
def make_pairgrid(d):
  """
    sns.PairGrid plotter for cycling data
    in: (d1,d2,d3,d4)
  """
  df = pd.DataFrame(np.concatenate(d))
  cond_labels = d[0].shape[0]*['fw top'] + d[1].shape[0]*['fw bot'] + d[2].shape[0]*['bw top'] + d[3].shape[0]*['bw bot']
  df['condition'] = cond_labels
  g = sns.PairGrid(df, hue='condition', diag_sharey=True)
  g.map_diag(plt.hist)
  g.map_offdiag(plt.plot)
  g.add_legend()
  return g

def plot_eigs(A_mat):
  """
    Docstring
  """
  w, _ = np.linalg.eig(A_mat)
  re_w = np.real(w)
  im_w = np.imag(w)
  f = plt.figure(figsize=(10, 10))
  plt.plot(re_w, im_w, 'o', alpha=0.9)
  theta = np.linspace(0, 2*np.pi, num=50)
  x_cir = np.cos(theta)
  y_cir = np.sin(theta)
  plt.plot(x_cir, y_cir, 'k', linewidth=0.5, alpha=0.5)
  plt.plot([-100, 100], [0, 0], 'k', linewidth=0.5, alpha=0.5)
  plt.plot([0, 0], [-100, 100], 'k', linewidth=0.5, alpha=0.5)
  plt.xlim([-1.5, 1.5])
  plt.ylim([-1.5, 1.5])
  return f

### PREPROCESSING
def parameter_grid_split(param_grid):
  """
    Takes a parameter grid dict, creates a list of dicts,
    where each new dict (in the list) varies only along one of its
    entries. Thus, instead of exhaustive grid search we are just 
    searching along individual hyperparameter axes.

    Probably a more elegant way to implement this without so many control flow
    statements.
  """
  keys = sorted(param_grid.keys())
  
  # Initialize first entry
  grid_out = [copy.deepcopy(param_grid)]
  for l in range(len(param_grid)):
    grid_out[0][keys[l]] = [param_grid[keys[l]][0]]

  # add new entries
  for l in range(len(param_grid)):
    if len(param_grid[keys[l]]) > 1:
      new_grid = copy.deepcopy(param_grid)
      for k in range(len(new_grid)):
        if k != l:
          new_grid[keys[k]] = [param_grid[keys[k]][0]]
        else:
          new_grid[keys[k]] = param_grid[keys[k]][1:]
      grid_out.append(new_grid)
  return grid_out

def preprocess_array(array, alpha=0):
  """
    Basic preprocessing.
    TODO: remove reshape/transpose. just save _preprocessed data
    use this function to normalize...
  """
  array = np.reshape(array, array.shape[:2] + (4,)) # order = 'C' or 'F'
  array = np.transpose(array, [1, 2, 0])

  # Normalize array
  max_ = np.max(array, axis=(0, 1))
  min_ = np.min(array, axis=(0, 1))
  return (array - min_)/(max_ - min_ + alpha)

def get_time_axis(kin):
  """ 
    Raw data is sampled every 1ms.
    We want every 25ms to make BPTT easier.
    We also want movement-period only.
    Input: kinematic array, data['D'][0,0]['KIN']
    Output: time_axis, t_inds1, t_inds2
      time_axis: 
      t_inds1: cycle 1:2, 2:3, 3:4, 4:5, 5:6
      t_inds2: cycle 1.5:2.5, 2.5:3.5, 3.5:4.5, 4.5:5.5
  """
  kin = np.reshape(kin.mean(-1), kin.shape[:2]+(4,))
  kin = np.transpose(kin, [1, 2, 0])

  time_inds = signal.argrelmin(kin[:, 0, 0]**2)[0]
  time_inds = time_inds[time_inds > 1500]
  time_inds = time_inds[time_inds < 4500]
  time_inds = time_inds[1:-1] # Remove first and last ind.
  # cycle 1:2, 2:3, 3:4, 4:5, 5:6
  t_inds1 = time_inds[::2]
  # cycle 1.5:2.5, 2.5:3.5, 3.5:4.5, 4.5:5.5
  t_inds2 = time_inds[1::2]
  time_axis = np.arange(t_inds1[0], t_inds1[-1], 25) # sample every 25ms
  return time_axis, t_inds1, t_inds2

def augmented_data(emg_in, t_inds, period, tiles=10):
  """
    Calculates a 'canonical cycle' of movementby averaging a few cycles together
    Then concatenates the cycles in time.
    This just creates more data for the RNN to fit.
  """
  signal_out = (len(t_inds) - 1)*[None]
  for i in range(len(t_inds) - 1):
    signal_out[i] = emg_in[t_inds[i]:t_inds[i + 1]]
    signal_out[i] = signal.resample(signal_out[i], period)
  signal_out = np.stack(signal_out).mean(axis=0)
  return np.tile(signal_out, (tiles, 1, 1))

def create_input_array(shape_in):
  """
    create shape (T,c,m) inupt array
    First input is 1 for forward movement
    Second input is 1 for backward movement
  """
  u_out = np.zeros(shape_in[:2] + (2,))
  u_out[:, 0, 0] = 1
  u_out[:, 1, 0] = 1
  u_out[:, 2, 1] = 1
  u_out[:, 3, 1] = 1
  return u_out

### ANALYSES
def get_path_length(signal_in, filt_freq=None):
  """
    Get the total path length of signal_in. Sum of Euclidean distances of nearby points. Filtering with a butterworth filter is optional.
    signal_in: a (T, batch_size, n) array
    filt_freq: filter frequency for signal.butter
    Output: a (batch_size,) array of path lengths
  """
  if filt_freq is not None:
    from scipy import signal
    b, a = signal.butter(4, filt_freq)
    signal_in = signal.filtfilt(b, a, signal_in, axis=0)
  return np.sum(np.sum(np.diff(signal_in, axis=0)**2, axis=-1)**0.5, axis=0)

def get_curvature(signal_in):
  """
    Input: a (T, n) array
    Output: a (T,) array of curvature values
    
    Curvature value at, say, t=5 takes datapoints at t=4,5,6, 
    finds the circumscribed circle and corresponding radius of curvature.
    Curvature is 1/radius.

    [wikipedia link]

    There is likely a better method for getting curvature:
    e.g. fit a 2nd order polynomial to a few data points around t,
    then find osculating circle of the polynomial at t.
  """
  def cross_id1(a, b, c):
    """
      Not needed.
    """
    return np.dot(a, c)*b - np.dot(a, b)*c

  def cross_id2(a, b):
    """
      cross product identity for ||a x b||
    """
    return np.sqrt(np.linalg.norm(a)**2*np.linalg.norm(b)**2 - np.dot(a, b)**2)

  k = np.zeros(signal_in.shape[0])
  for t in range(1, signal_in.shape[0]-1):
    A, B, C = signal_in[t-1:t+2]
    r = np.linalg.norm(A)*np.linalg.norm(B)*np.linalg.norm(A-B)/(2*cross_id2(A, B))
    k[t] = 1/r

  # fix start and end values
  k[0] = k[1]
  k[-1] = k[-2]
  return k

def get_generalized_curvature(signal_in, total_points, deg):
  """
    Calculate curvature of n-dimensional trajectory, signal_in of shape (t, n)
    Use total_points adjacent points to fit a polynomial of degree deg
    Then calculate generalized curvature explicitly from the polynomial
    total_points: ~11
    deg: ~5
  """

  def dt_id1(a, ap):
    """ 
      derivative of a/norm(a)
      input: a and da/dt, a and da/dt are n-dim vectors
      output: d/dt ( a/norm(a) )
    """
    return ap/np.linalg.norm(a) - np.dot(ap, a)*a/(np.dot(a, a)**(1.5))

  def dt_id2(a, b, ap, bp):
    """ 
      derivative of inner(a, b)*b
      input: a, b, da/dt, db/dt, each an n-dim vector
      output: d/dt ( inner(a, b)*b )
    """
    return (np.dot(ap, b) + np.dot(a, bp))*b + np.dot(a, b)*bp

  total_curvatures = np.min((signal_in.shape[1]-1, deg-1)) # have many curvatures to calculate
  k_t = np.zeros((signal_in.shape[0], total_curvatures)) # initialize curvatures, k
  e_t = np.zeros(signal_in.shape+(total_curvatures+1,)) # frenet frames, (t, n, curvatures+1)

  half = np.floor(total_points/2).astype(int)

  for t in range(half, signal_in.shape[0] - half): # end point correct?
    times = np.arange(t-half, t+half+1)
    times_local = times - t # will always be -half:half
    tmid = times_local[half] # tmid = 0
    p = P.polyfit(times_local, signal_in[times], deg)
    pp = [] # coefficients of polynomial (and its derivatives)
    pt = [] # polynomial (and its derivs) evaluated at time t
    for deriv in range(deg+2): # +2 because there are deg+1 nonzero derivatives of the polynomial, and +1 because of range()
      pp.append(P.polyder(p, deriv))
      pt.append(P.polyval(tmid, pp[-1]).T) # evaluate at 0

    e = [] # frenet basis, e1, e2, ... at time t
    ep = [] # derivatives, e1', e2', ...
    e_ = [] # unnormalized es
    e_p = [] # unnormalized (e')s

    k = [] # generalized curvature at time t

    # first axis of frenet frame
    e.append(pt[1]/np.linalg.norm(pt[1]))
    ep.append(dt_id1(pt[1], pt[2]))

    for dim in range(2, total_curvatures+2):
      # Start gram-schmidt orthogonalization on e:
      e_.append(pt[dim])
      e_p.append(pt[dim+1])
      for j in range(dim-1):
        e_[-1] = e_[-1] - np.dot(pt[dim], e[j])*e[j] # orthogonalize relative to every other e
        e_p[-1] = e_p[-1] - dt_id2(pt[dim], e[j], pt[dim+1], ep[j]) # derivative of e_
      e.append(e_[-1]/np.linalg.norm(e_[-1])) # normalize e_ to get e
      ep.append(dt_id1(e_[-1], e_p[-1])) # derivative of e

      k.append(np.dot(ep[-2], e[-1])/np.linalg.norm(pt[1]))
    k_t[t, :] = np.array(k)
    e_t[t, :, :] = np.array(e).T

  return k_t, e_t

def run_rnn(monkey='D',
            beta1=0.0,
            beta2=0.0,
            stddev_state=0.0,
            stddev_out=0.0,
            activation='tanh',
            num_neurons=100,
            learning_rate=0.0001,
            num_iters=2000,
            load_prev=False,
            save_model_path='./saves/',
            load_model_path=None,
            tb_path='./tensorboard/',
            local_machine=True):
  """
    monkey: 'D' or 'C'
    beta1: regularization hyperparameter for l2_loss(A)
    beta2: regularization hyperparameter for l2_loss(C)
    stddev_state: stddev of injected noise in state variable
    stddev_out: stddev of injected noise in output
    activation: nonlinearity for the RNN. use lambda x: x for linear.
    num_neurons: state dimension
    learning_rate: learning rate for Adam
    num_iters: training iterations
    load_prev: whether or not to load the previous TF variables
    save_model_path: where to save the TF model using tf.train.Saver()
    load_model_path: If load_prev=True, where to load the previous model
    tb_path: tensorboard path
    local_machine: is this a local machine or cluster run?
  """
  if local_machine:
    path_prefix = '/Users/jeff/Documents/Python/_projects/cyclingRNN/'
  else:
    path_prefix = '/vega/zmbbi/users/jss2219/cyclingRNN/'

  # TODO: just load *_preprocessed.mat data.
  if monkey == 'D':
    data = sio.loadmat(path_prefix+'drakeFeb.mat')
  else:
    data = sio.loadmat(path_prefix+'cousFeb.mat')

  # Set activation
  if activation == 'tanh':
    activation = tf.tanh
  elif activation == 'linear':
    activation = lambda x: x

  # Preprocess data
  emg = preprocess_array(data['D'][0, 0]['EMG'])
  time_axis, time_inds1, time_inds2 = get_time_axis(data['D'][0, 0]['KIN'])
  y_data1 = emg[time_axis]
  p = y_data1.shape[-1]

  # Build inputs
  m = 2
  u_data1 = create_input_array(y_data1.shape)

  # Augmented data
  # For regularizing the network -- it must fit actual and augmented data
  period = int(np.round(np.diff(time_inds2).mean()))
  y_cat1 = augmented_data(emg, time_inds1, period=period, tiles=10)
  y_cat1 = y_cat1[::25]
  y_cat2 = augmented_data(emg, time_inds2, period=period, tiles=10)
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

  # Tensorflow graph
  tf.reset_default_graph()

  n = num_neurons
  batch_size = y_data.shape[1]

  x0 = tf.Variable(tf.random_normal([batch_size, n], stddev=0.1), name='x0')

  C = tf.Variable(tf.random_normal([n, p], stddev=1/np.sqrt(n)), name='C')
  d = tf.Variable(tf.constant(0.01, shape=[1, p]), name='d')

  U = tf.placeholder(tf.float32, [None, batch_size, m], name='U')
  Y = tf.placeholder(tf.float32, [None, batch_size, p], name='Y')

  noise_state = tf.placeholder(tf.float32, name='stddev_state')

  time_steps = tf.shape(U)[0]

  # BasicRNNCellNoise - inelegant solution to adding state noise to RNNs. but not sure of an elegant method.
  # TODO: scale cell noise by num_neurons... 1/sqrt(n)? 
  cell = BasicRNNCellNoise(n, activation=activation, stddev=noise_state, batch_size=batch_size)
  output, state = tf.nn.dynamic_rnn(cell, U, initial_state=x0, dtype=tf.float32, time_major=True)

  print output.name
  Y_hat = tf.reshape(output, (time_steps*batch_size, n))
  Y_hat = tf.matmul(Y_hat, C) + d
  Y_hat = tf.reshape(Y_hat, (time_steps, batch_size, p), name='Y_hat')

  # Get RNN variables
  with tf.variable_scope('RNN/BasicRNNCellNoise/Linear', reuse=True):
    Mat = tf.get_variable('Matrix', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1/np.sqrt(n)))
    A = tf.gather(tf.get_variable('Matrix'), range(m, m+n))
    B = tf.gather(tf.get_variable('Matrix'), range(0, m))
    b = tf.get_variable('Bias')

  # Training ops
  cost_term1 = tf.reduce_mean((Y_hat - Y)**2, name='cost1')
  cost_term2 = beta1*tf.nn.l2_loss(A)
  cost_term3 = beta2*tf.nn.l2_loss(C)
  cost = cost_term1 + cost_term2 + cost_term3
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
  opt_op = train_op.minimize(cost)

  # Summary ops
  tf.scalar_summary('loss', cost)
  tf.scalar_summary('log_loss', tf.log(cost))
  tf.scalar_summary('cost 1', cost_term1)
  tf.scalar_summary('cost 2', cost_term2)
  tf.scalar_summary('cost 3', cost_term3)

  for var in tf.all_variables():
    tf.histogram_summary(var.name, var)
  merged_summary_op = tf.merge_all_summaries()

  # Saver ops
  saver = tf.train.Saver()

  # Train
  # TODO: add stopping criterion
  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(tb_path, graph=sess.graph)
    sess.run(tf.initialize_all_variables())
    if load_prev and os.path.exists(load_model_path):
      saver.restore(sess, load_model_path)

    for i in range(num_iters):
      feed_dict = {Y: y_data + np.random.randn(*y_data.shape)*y_data.var()*stddev_out,
                   U: u_data,
                   noise_state: stddev_state}
      _, loss_val, summary_str = sess.run([opt_op, cost, merged_summary_op], feed_dict=feed_dict)

      if i % 50 == 0:
        summary_writer.add_summary(summary_str, i)

      if i % 500 == 0:
        print '  iter:', '%04d' % (i), \
              '  Loss:', '{:.6f}'.format(loss_val)
        saver.save(sess, save_model_path)

    print '  iter:', '%04d' % (num_iters), \
          '  Loss:', '{:.6f}'.format(loss_val)
    saver.save(sess, save_model_path)

    print '  Finished'

    # Simulate
    y_tf, x_tf = sess.run([Y_hat, output], feed_dict=feed_dict)

  return y_tf, x_tf
