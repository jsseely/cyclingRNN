"""
  A collection of functions for the 'cycling RNN' project.
  run_rnn is the main function that builds the tensorflow graph and does the training.
"""
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy import signal
from cyclingrnn.custom_rnn_cells import BasicRNNCellNoise

def preprocess_array(array, alpha=0):
  """
    Basic preprocessing.
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
    apply butter filter to smooth the edges.
    there are better ways to do this. but time in life is limited.
    This just creates more data for the RNN to fit.
  """
  signal_out = (len(t_inds) - 1)*[None]
  for i in range(len(t_inds) - 1):
    signal_out[i] = emg_in[t_inds[i]:t_inds[i + 1]]
    signal_out[i] = signal.resample(signal_out[i], period)
  signal_out = np.stack(signal_out).mean(axis=0)
  b, a = signal.butter(3, 0.020) # change if dt chnages
  signal_out = np.tile(signal_out, (tiles, 1, 1)) 
  return signal.filtfilt(b, a, signal_out, axis=0)

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

def train_rnn(monkey='D',
              beta1=0.0,
              beta2=0.0,
              stddev_state=0.0,
              stddev_out=0.0,
              activation='tanh',
              rnn_init='orth',
              num_neurons=100,
              learning_rate=0.0001,
              num_iters=2000,
              save_model_path='./saves/',
              tb_path='./tensorboard/',
              load_prev=False,
              load_model_path=None):
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

  # TODO: just load *_preprocessed.mat data.
  if monkey == 'D':
    data = sio.loadmat('./drakeFeb.mat') #TODO: fix, '../' or './' depending on whether running from wrapper or not
  elif monkey == 'C':
    data = sio.loadmat('./cousFeb.mat')

  # Set activation
  if activation == 'tanh':
    activation = tf.tanh
  elif activation == 'linear':
    activation = tf.identity
  elif activation == 'softplus':
    activation = tf.nn.softplus

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

  total_data_points = np.sum([v*4 for v in sequence_length])

  # Tensorflow graph
  tf.reset_default_graph()
  #tf.set_random_seed(1234)

  n = num_neurons
  batch_size = y_data.shape[1]

  x0 = tf.Variable(tf.random_normal([batch_size, n], stddev=0.1), name='x0')

  C = tf.get_variable('C', shape=[n, p], initializer=tf.contrib.layers.xavier_initializer())
  #C = tf.Variable(tf.random_normal([n, p], stddev=1/np.sqrt(n)), name='C')
  d = tf.get_variable('d', shape=[1, p], initializer=tf.constant_initializer(0))
  #d = tf.Variable(tf.constant(0.01, shape=[1, p]), name='d')

  U = tf.placeholder(tf.float32, [u_data.shape[0], batch_size, m], name='U')
  Y = tf.placeholder(tf.float32, [y_data.shape[0], batch_size, p], name='Y')

  noise_state = tf.placeholder(tf.float32, name='stddev_state')

  time_steps = tf.shape(U)[0]

  # set initializer for rnn matrix
  if rnn_init == 'orth':
    rnn_initializer = tf.orthogonal_initializer(0.95)
  elif rnn_init == 'xavier':
    rnn_initializer = tf.contrib.layers.xavier_initializer()
  elif rnn_init == 'normal':
    rnn_initializer = tf.random_normal_initializer(1/np.sqrt(n))

  # get a tf var scope to set the rnn initializer. 
  with tf.variable_scope('RNN', initializer=rnn_initializer) as scope:
    pass

  cell = tf.nn.rnn_cell.BasicRNNCell(n, activation=activation)
  output, state = tf.nn.dynamic_rnn(cell, U, sequence_length=4*[sequence_length[0]]+4*[sequence_length[1]]+4*[sequence_length[2]], initial_state=x0, dtype=tf.float32, time_major=True, scope=scope)

  Y_hat = tf.reshape(output, (time_steps*batch_size, n))
  Y_hat = tf.matmul(Y_hat, C) + d
  Y_hat = tf.reshape(Y_hat, (time_steps, batch_size, p), name='Y_hat')

  # Get RNN variables
  with tf.variable_scope('RNN/BasicRNNCell/Linear', reuse=True):
    Mat = tf.get_variable('Matrix') #note: calling an initializer here will not give it a new one. 
    A = tf.gather(tf.get_variable('Matrix'), range(m, m+n))
    B = tf.gather(tf.get_variable('Matrix'), range(0, m))
    b = tf.get_variable('Bias')

  # Training ops
  # take L2 loss only over data points. note that dynamic_rnn zeros out output, but not y_hat because we have the bias vector d
  cost_term1 = 0.5*tf.reduce_sum((Y_hat[:sequence_length[0], :4, :] - Y[:sequence_length[0], :4, :])**2)
  cost_term1 += 0.5*tf.reduce_sum((Y_hat[:sequence_length[1], 4:8, :] - Y[:sequence_length[1], 4:8, :])**2)
  cost_term1 += 0.5*tf.reduce_sum((Y_hat[:sequence_length[2], 8:, :] - Y[:sequence_length[2], 8:, :])**2)
  cost_term1 = cost_term1/total_data_points

  cost_term2 = beta1*0.5*tf.reduce_sum(A**2)
  cost_term3 = beta2*0.5*tf.reduce_sum(C**2)
  cost = cost_term1 + cost_term2 + cost_term3

  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
  opt_op = train_op.minimize(cost)

  # Summary ops
  tf.summary.scalar('loss', cost)
  tf.summary.scalar('log_loss', tf.log(cost))
  tf.summary.scalar('cost_1', cost_term1)
  tf.summary.scalar('cost_2', cost_term2)
  tf.summary.scalar('cost_3', cost_term3)

  merged_summary_op = tf.summary.merge_all()

  # Saver ops
  saver = tf.train.Saver()

  # Train
  with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    # TODO: fix restore. new tf version saves files differently?
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

    print '  iter:', '%04d' % (num_iters), \
          '  Loss:', '{:.6f}'.format(loss_val)
    saver.save(sess, save_model_path)

    print '  Finished'

    # Simulate
    y_tf, x_tf = sess.run([Y_hat, output], feed_dict=feed_dict)

  return y_tf, x_tf
