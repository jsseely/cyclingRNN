import numpy as np
import tensorflow as tf
import scipy.io as sio
from scipy.io import savemat
import datetime

def run_rnn(monkey='D', beta1=0.0, beta2=0.0, learning_rate=0.0001, num_iters=2000, load_prev=False, load_model_path=''):

  if monkey=='D':
    data = sio.loadmat('/Users/jeff/Documents/Python/_projects/cyclingRNN/drakeFeb.mat')
  else:
    data = sio.loadmat('/Users/jeff/Documents/Python/_projects/cyclingRNN/cousFeb.mat')
  
  m1 = data['D'][0,0]['M1']
  emg = data['D'][0,0]['EMG']

  m1 = np.reshape(m1, m1.shape[:2]+(4,)) # order = 'C' or 'F'
  emg = np.reshape(emg, emg.shape[:2]+(4,)) # order = 'C' or 'F'

  m1 = np.transpose(m1, [1,2,0])
  emg = np.transpose(emg, [1,2,0])
  # Preprocess data
  # Normalize EMG
  max_ = np.max(emg, axis=(0,1))
  min_ = np.min(emg, axis=(0,1))
  emg_ = (emg - min_)/(max_ - min_)

  # Normalize M1
  max_ = np.max(m1, axis=(0,1))
  min_ = np.min(m1, axis=(0,1))
  m1_ = m1/(max_ - min_ + 5)

  # Select times + downsample 
  times = np.arange(2000,4000, 25)
  m1_ = m1[times]
  emg_ = emg_[times]

  # set up data for TF
  time_steps = emg_.shape[0]
  y_data = emg_
  m = 2
  u_data = np.zeros(y_data.shape[:2]+(m,))
  u_data[:,0,0] = 1
  u_data[:,1,0] = 1
  u_data[:,2,1] = 1
  u_data[:,3,1] = 1

  ## TF part
  tf.reset_default_graph()

  n = 100 # n = 100 neurons
  p = y_data.shape[-1] # p = 36 muscles
  total_batches = y_data.shape[1]

  x0 = tf.Variable(tf.random_normal([total_batches,n], stddev=0.01), name='x0')

  C = tf.Variable(tf.random_normal([n,p], stddev=1/np.sqrt(n)), name='C')
  d = tf.Variable(tf.constant(0.01, shape=[1,p]), name='d')

  B = tf.Variable(tf.random_normal([m,n], stddev=1/np.sqrt(n)), name='B')

  U = tf.placeholder(tf.float32, [time_steps, None, m], name='U')
  Y = tf.placeholder(tf.float32, [time_steps, None, p], name='Y')

  U_ = tf.unpack(U)
  U_ = [tf.matmul(U_[i], B) for i in range(time_steps)]
  U_ = tf.pack(U_)

  cell = tf.nn.rnn_cell.BasicRNNCell(n)
  #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
  output, state = tf.nn.dynamic_rnn(cell, U_, initial_state=x0, dtype=tf.float32, time_major=True)

  Y_hat = tf.unpack(output)
  Y_hat = [tf.matmul(Y_hat[i], C) + d for i in range(time_steps)]
  Y_hat = tf.pack(Y_hat)

  #Get A matrix
  with tf.variable_scope('RNN/BasicRNNCell/Linear', reuse=True):
    A = tf.get_variable('Matrix')
    b = tf.get_variable('Bias')

  cost = tf.reduce_mean((Y_hat - Y)**2)+beta1*tf.nn.l2_loss(A)+beta2*tf.nn.l2_loss(C) 
          
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate) # 0.0001 
  opt_op = train_op.minimize(cost)

  ##
  tf.scalar_summary('loss', cost)
  tf.histogram_summary('A', A)
  tf.histogram_summary('B', B)
  tf.histogram_summary('b', b)
  tf.histogram_summary('C', C)
  tf.histogram_summary('d', d)
  tf.histogram_summary('x0', x0)
  merged_summary_op = tf.merge_all_summaries()

  saver = tf.train.Saver()
  cur_run = monkey+'_'+str(datetime.datetime.now().strftime("%m%d-%H%M-%S"))
  load_model_path = './saves/'+cur_run
  save_model_path = './saves/'+cur_run

  print "Current run:"+cur_run

  ## Train:
  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs/'+cur_run, graph=sess.graph)
    sess.run(tf.initialize_all_variables())
    if load_prev:
      saver.restore(sess, load_model_path) 

    for i in range(num_iters):
      feed_dict = {Y: y_data, U: u_data}
      _, loss_val, summary_str = sess.run([opt_op, cost, merged_summary_op], feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

      if i % 500 == 0:
        print 'iter:', '%04d' % (i), \
              'Loss:', '{:.6f}'.format(loss_val)
        saver.save(sess, save_model_path)

    print 'iter:', '%04d' % (i), \
          'Loss:', '{:.6f}'.format(loss_val)
    saver.save(sess, save_model_path)

    print 'Finished'

    # simulate model
    y_tf, x_tf, loss_val = sess.run([Y_hat, output, cost], feed_dict=feed_dict)

  return y_tf, x_tf, loss_val

