import datetime
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy import signal

def preprocess_array(array, alpha=0):
  """
    Basic preprocessing.
  """
  array = np.reshape(array, array.shape[:2] + (4,)) # order = 'C' or 'F'
  array = np.transpose(array, [1, 2, 0])

  # Normalize array
  max_ = np.max(array, axis=(0, 1))
  min_ = np.min(array, axis=(0, 1))
  return (array - min_)/(max_ - min_ + alpha)

def get_time_axis(kin):
  """ 
    Input: kinematic array, data['D'][0,0]['KIN']
    Output: time_axis, t_inds1, t_inds2
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
  """
  signal_out = (len(t_inds) - 1)*[None]
  for i in range(len(t_inds) - 1):
    signal_out[i] = emg_in[t_inds[i]:t_inds[i + 1]]
    signal_out[i] = signal.resample(signal_out[i], period)
  signal_out = np.stack(signal_out).mean(axis=0)
  return np.tile(signal_out, (tiles, 1, 1))

def create_input_array(shape_in):
  """
  """
  u_out = np.zeros(shape_in[:2] + (2,))
  u_out[:, 0, 0] = 1
  u_out[:, 1, 0] = 1
  u_out[:, 2, 1] = 1
  u_out[:, 3, 1] = 1
  return u_out

def run_rnn(monkey='D',
            beta1=0.0,
            beta2=0.0,
            activation=tf.tanh,
            num_neurons=100,
            learning_rate=0.0001,
            num_iters=2000,
            load_prev=False,
            save_model_path='./saves/',
            load_model_path=None,
            tb_path='./tensorboard/',
            local_machine=True):

  if local_machine:
    path_prefix = '/Users/jeff/Documents/Python/_projects/cyclingRNN/'
  else:
    path_prefix = '/vega/zmbbi/users/jss2219/cyclingRNN/'

  if monkey == 'D':
    data = sio.loadmat(path_prefix+'drakeFeb.mat')
  else:
    data = sio.loadmat(path_prefix+'cousFeb.mat')

  # Preprocess data
  emg = preprocess_array(data['D'][0, 0]['EMG'])
  time_axis, time_inds1, time_inds2 = get_time_axis(data['D'][0, 0]['KIN'])
  y_data1 = emg[time_axis]
  p = y_data1.shape[-1]

  # build inputs
  m = 2
  u_data1 = create_input_array(y_data1.shape)

  ###### Augmented data
  period = int(np.round(np.diff(time_inds2).mean()))
  y_cat1 = augmented_data(emg, time_inds1, period=period, tiles=10)
  y_cat1 = y_cat1[::25]
  y_cat2 = augmented_data(emg, time_inds2, period=period, tiles=10)
  y_cat2 = y_cat2[::25]

  u_cat1 = create_input_array(y_cat1.shape)
  u_cat2 = create_input_array(y_cat2.shape)
  ######

  ###### Sequences
  sequence_length = [y_data1.shape[0], y_cat1.shape[0], y_cat2.shape[0]]
  y_data = np.zeros((np.max(sequence_length), 4*3, p))
  u_data = np.zeros((np.max(sequence_length), 4*3, m))

  y_data[:sequence_length[0], 0:4, :] = y_data1
  y_data[:sequence_length[1], 4:8, :] = y_cat1
  y_data[:sequence_length[2], 8:12, :] = y_cat2

  u_data[:sequence_length[0], 0:4, :] = u_data1
  u_data[:sequence_length[1], 4:8, :] = u_cat1
  u_data[:sequence_length[2], 8:12, :] = u_cat2
  ######

  ## TF part
  tf.reset_default_graph()

  n = num_neurons
  batch_size = y_data.shape[1]

  x0 = tf.Variable(tf.random_normal([batch_size, n], stddev=0.1), name='x0')

  C = tf.Variable(tf.random_normal([n, p], stddev=1/np.sqrt(n)), name='C')
  d = tf.Variable(tf.constant(0.01, shape=[1, p]), name='d')

  U = tf.placeholder(tf.float32, [None, batch_size, m], name='U')
  Y = tf.placeholder(tf.float32, [None, batch_size, p], name='Y')

  time_steps = tf.shape(U)[0]

  cell = tf.nn.rnn_cell.BasicRNNCell(n, activation=activation)
  #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
  output, state = tf.nn.dynamic_rnn(cell, U, initial_state=x0, dtype=tf.float32, time_major=True)

  Y_hat = tf.reshape(output, (time_steps*batch_size, n))
  Y_hat = tf.matmul(Y_hat, C) + d
  Y_hat = tf.reshape(Y_hat, (time_steps, batch_size, p))

  with tf.variable_scope('RNN/BasicRNNCell/Linear', reuse=True):
    Mat = tf.get_variable('Matrix', initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1/np.sqrt(n)))
    A = tf.gather(tf.get_variable('Matrix'), range(m, m+n))
    B = tf.gather(tf.get_variable('Matrix'), range(0, m))
    b = tf.get_variable('Bias')

  # Training ops
  cost_term1 = tf.reduce_mean((Y_hat - Y)**2)
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
      # TODO: is the noise here necessary? What is the right scaling?
      feed_dict = {Y: y_data + np.random.randn(*y_data.shape)*y_data.var()*0.1,
                   U: u_data}
      _, loss_val, summary_str = sess.run([opt_op, cost, merged_summary_op], feed_dict=feed_dict)

      if i % 10 == 0:
        summary_writer.add_summary(summary_str, i)

      if i % 1000 == 0:
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