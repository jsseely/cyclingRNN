""" Cycling RNN """

import numpy as np
import tensorflow as tf
import datetime
from scipy.io import savemat

class RNN(object):
  def __init__(self, input_dim, state_dim, output_dim, time_steps, num_sequences, seq_length, monkey='D', beta1=0, beta2=0):
    """
    """
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.output_dim = output_dim
    self.time_steps = time_steps
    self.num_sequences = num_sequences
    self.seq_length = seq_length
    self.monkey = monkey
    self.beta1 = beta1
    self.beta2 = beta2

  def build_graph(self):
    # initial state
    x0 = tf.Variable(tf.random_normal([self.num_sequences, self.state_dim], stddev=0.01), name='x0')

    C = tf.Variable(tf.random_normal([self.state_dim,self.output_dim], stddev=1/np.sqrt(self.state_dim)), name='C')
    d = tf.Variable(tf.constant(0.01, shape=[1,self.output_dim]), name='d')

    B = tf.Variable(tf.random_normal([self.input_dim,self.state_dim], stddev=1/np.sqrt(self.state_dim)), name='B')

    U = tf.placeholder(tf.float32, [self.time_steps, None, self.input_dim], name='U')
    Y = tf.placeholder(tf.float32, [self.time_steps, None, self.output_dim], name='Y')

    U_ = tf.unpack(U)
    U_ = [tf.matmul(U_[i], B) for i in range(self.time_steps)]
    U_ = tf.pack(U_)

    cell = tf.nn.rnn_cell.BasicRNNCell(self.state_dim)
    #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
    output, state = tf.nn.dynamic_rnn(cell, U_, initial_state=x0, dtype=tf.float32, time_major=True)

    Y_hat = tf.unpack(output)
    Y_hat = [tf.matmul(Y_hat[i], C) + d for i in range(self.time_steps)]
    Y_hat = tf.pack(Y_hat)

    output_diff = tf.gather(output, range(1,time_steps))-tf.gather(output, range(0,time_steps-1))

    #Get A matrix
    with tf.variable_scope('RNN/BasicRNNCell/Linear', reuse=True):
      A = tf.get_variable('Matrix')
      b = tf.get_variable('Bias')

    # Loss
    cost = tf.reduce_mean((Y_hat - Y)**2) + self.beta1*tf.nn.l2_loss(A) + self.beta2*tf.nn.l2_loss(C)

    # trainers / optimizers  
    train_op = tf.train.AdamOptimizer(learning_rate=0.001) # add gradient_noise ? 
    #grads_and_vars = train_op.compute_gradients(cost)
    opt_op = train_op.minimize(cost)

    # summaries
    tf.scalar_summary('loss', cost)
    tf.histogram_summary('A', A)
    tf.histogram_summary('B', B)
    tf.histogram_summary('b', b)
    tf.histogram_summary('C', C)
    tf.histogram_summary('d', d)
    tf.histogram_summary('x0', x0)
    merged_summary_op = tf.merge_all_summaries()

    # saver ops
    saver = tf.train.Saver()
    cur_run = monkey+'_'+str(datetime.datetime.now().strftime("%m%d-%H%M-%S"))
    load_model_path = './saves/'+cur_run
    save_model_path = './saves/'+cur_run

  def train(self, steps, restore=False):
    with tf.Session() as sess:
      summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs/'+cur_run, graph=sess.graph)
      sess.run(tf.initialize_all_variables())
      if restore:
        saver.restore(sess, load_model_path) # comment out if not loading

      for i in range(steps):
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
    return sess.run([Y_hat, output], feed_dict=feed_dict)
