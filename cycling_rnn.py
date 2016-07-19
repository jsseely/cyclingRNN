import datetime
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy import signal


  # def get_canon(inds_, data_in):
  #   """
  #     get canonical emg data.
  #     approach is a bit complicated...
  #   """
  #   can_out = (len(inds_)-1)*[None]
  #   for i in range(len(inds_)-1):
  #     can_out[i] = data_in[inds_[i]:inds_[i+1]]
  #     period = int(np.round(np.diff(inds_).mean()))
  #     can_out[i] = signal.resample(can_out[i], period)
  #   can_out = np.stack(can_out).mean(axis=0)
  #   return np.tile(can_out, (10, 1, 1)), period

  # # canon1 
  # canon1, p1 = get_canon(time_inds1, data_dict['D'][0, 0]['EMG'])
  # canon2, p2 = get_canon(time_inds2, data_dict['D'][0, 0]['EMG'])
  # canon = np.mean(np.stack([canon1[p1/2:], canon2[:-p1/2+1]]), axis=0)
  # return canon

def run_rnn(monkey='D',
            beta1=0.0,
            beta2=0.0,
            learning_rate=0.0001,
            num_iters=2000,
            load_prev=False,
            save_model_path='./saves/',
            load_model_path=None,
            tb_path='./tensorboard/',
            local_machine=True):

  if local_machine:
    path_prefix = '/Users/jeff/Documents/Python/_projects/cyclingRNN/'
    # tb_prefix = '/tmp/tf/'
  else:
    path_prefix = '/vega/zmbbi/users/jss2219/cyclingRNN/'
    # tb_prefix = '/vega/zmbbi/users/jss2219/cyclingRNN/tensorboard/'

  if monkey == 'D':
    data = sio.loadmat(path_prefix+'drakeFeb.mat')
  else:
    data = sio.loadmat(path_prefix+'cousFeb.mat')

  emg = data['D'][0, 0]['EMG']
  emg = np.reshape(emg, emg.shape[:2]+(4, )) # order = 'C' or 'F'
  emg = np.transpose(emg, [1, 2, 0])

  kin = data['D'][0, 0]['KIN']
  kin = np.reshape(kin.mean(-1), kin.shape[:2]+(4,))
  kin = np.transpose(kin, [1, 2, 0])

  # Get time indices
  time_inds = signal.argrelmin(kin[:,0,0]**2)[0]
  time_inds = time_inds[time_inds > 1500]
  time_inds = time_inds[time_inds < 4500]
  time_inds1 = time_inds[:-1:2]
  time_inds2 = time_inds[1::2]

  time_axis = np.arange(time_inds2[0], time_inds2[-1], 25)

  # Preprocess data
  # Normalize EMG
  max_ = np.max(emg, axis=(0, 1))
  min_ = np.min(emg, axis=(0, 1))
  emg_ = (emg - min_)/(max_ - min_)

  # Select times + downsample 
  emg_ = emg_[time_axis]

  # set up data for TF
  time_steps = emg_.shape[0]
  y_data = emg_

  m = 2
  u_data = np.zeros(y_data.shape[:2]+(m, ))
  u_data[:, 0, 0] = 1
  u_data[:, 1, 0] = 1
  u_data[:, 2, 1] = 1
  u_data[:, 3, 1] = 1

  ######
  def augment_data():
    """
      return a canonical emg
    """
    can_out = (len(time_inds2)-1)*[None]
    for i in range(len(time_inds2)-1):
      can_out[i] = emg[time_inds2[i]:time_inds2[i+1]]
      period = int(np.round(np.diff(time_inds2).mean()))
      can_out[i] = signal.resample(can_out[i], period)
    can_out = np.stack(can_out).mean(axis=0)
    return np.tile(can_out, (10, 1, 1))

  # Get 'canonical' EMG data
  # TODO: simplify
  y_cat = augment_data()
  y_cat = y_cat[::25]
  u_cat = np.zeros(y_cat.shape[:2]+(m, ))
  u_cat[:, 0, 0] = 1
  u_cat[:, 1, 0] = 1
  u_cat[:, 2, 1] = 1
  u_cat[:, 3, 1] = 1
  ######

  ## TF part
  tf.reset_default_graph()

  n = 100 # n = 100 neurons
  p = y_data.shape[-1] # p = 36 muscles
  batch_size = y_data.shape[1]

  x0 = tf.Variable(tf.random_normal([batch_size, n], stddev=0.1), name='x0')

  C = tf.Variable(tf.random_normal([n, p], stddev=1/np.sqrt(n)), name='C')
  d = tf.Variable(tf.constant(0.01, shape=[1, p]), name='d')

  U = tf.placeholder(tf.float32, [time_steps, None, m], name='U')
  Y = tf.placeholder(tf.float32, [time_steps, None, p], name='Y')

  cell = tf.nn.rnn_cell.BasicRNNCell(n)
  #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
  output, state = tf.nn.dynamic_rnn(cell, U, initial_state=x0, dtype=tf.float32, time_major=True)

  Y_hat = tf.unpack(output)
  Y_hat = [tf.matmul(Y_hat[i], C) + d for i in range(time_steps)]
  Y_hat = tf.pack(Y_hat)

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
  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(tb_path, graph=sess.graph)
    sess.run(tf.initialize_all_variables())
    if load_prev and os.path.exists(load_model_path):
      saver.restore(sess, load_model_path)

    for i in range(num_iters):
      # TODO: is the noise here necessary? What is the right scaling?
      feed_dict = {Y: y_data + np.random.randn(*y_data.shape)*y_data.var()*0.2,
                   U: u_data}
      _, loss_val, summary_str = sess.run([opt_op, cost, merged_summary_op], feed_dict=feed_dict)
      
      # Train on entire sequence
      sess.run(opt_op, feed_dict={Y: y_cat + np.random.randn(*y_cat.shape)*y_cat.var()*0.2, U: u_cat})

      if i % 10 == 0:
        summary_writer.add_summary(summary_str, i)

      if i % 500 == 0:
        print '  iter:', '%04d' % (i), \
              '  Loss:', '{:.6f}'.format(loss_val)
        saver.save(sess, save_model_path)

    print '  iter:', '%04d' % (i), \
          '  Loss:', '{:.6f}'.format(loss_val)
    saver.save(sess, save_model_path)

    print '  Finished'

    # Simulate
    y_tf, x_tf, loss_val = sess.run([Y_hat, output, cost], feed_dict=feed_dict)

  return y_tf, x_tf, loss_val


# def save_to_matlab(load_model_path=None):
#   #TODO: fix this.
#   # Save RNN
#   savestr = '/Users/jeff/Documents/MATLAB/CyclingTask/data/tf_'+cur_run+'.mat'
#   sio.savemat(savestr, mdict={'X': x_tf})
#   savestr = '/Users/jeff/Documents/MATLAB/CyclingTask/data/tf_'+cur_run+'_params.mat'
#   #TODO: better way to implement this...
#   sio.savemat(savestr, mdict={'A': A_tf,
#                               'B': B_tf,
#                               'C': C_tf,
#                               'b': b_tf,
#                               'd': d_tf,
#                               'x0': x0_tf,
#                               'monkey': monkey,
#                               'beta1': beta1,
#                               'beta2': beta2,
#                               'learning_rate': learning_rate,
#                               'num_iters': num_iters,
#                               'load_prev': load_prev,
#                               'load_model_path': load_model_path})
