""" Cycling RNN """

import numpy as np
import tensorflow as tf

class RNN(object):
  def __init__(self, input_dim, state_dim, output_dim, time_steps, num_sequences, seq_length):
    """
    """
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.output_dim = output_dim
    self.time_steps = time_steps
    self.num_sequences = num_sequences
    self.seq_length = seq_length

    # Placeholders=
    U = tf.placeholder(tf.float32, [self.time_steps, None, self.n], name='U')
    Y = tf.placeholder(tf.float32, [self.time_steps, None, self.p], name='Y')
    x0_in = tf.placeholder(tf.float32, [None, ], name='x0_in')

    # State sequences.
    # TODO: implement num_sequences.
    x0 = tf.Variable(tf.random_normal([self.state_dim, self.output_dim], stddev=0.01), name='x0')

    C = tf.Variable(tf.random_normal([self.state_dim,self.output_dim], stddev=1/np.math.sqrt(self.state_dim)), name='C')
    d = tf.Variable(tf.constant(0.01, shape=[1,self.output_dim]), name='d')

    cell = tf.nn.rnn_cell.BasicRNNCell(self.state_dim)
    #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
    output, state = tf.nn.dynamic_rnn(cell, U, initial_state=x0, dtype=tf.float32, time_major=True)

    Y_hat = tf.unpack(output)
    Y_hat = [tf.matmul(Y_hat[i], C) + d for i in range(time_steps)]
    Y_hat = tf.pack(Y_hat)

    # X_{t} - X_{t-1}
    state_diff = tf.gather(output, range(1,time_steps))-tf.gather(output, range(0,time_steps-1))

    #Get A matrix
    with tf.variable_scope('RNN/BasicRNNCell/Linear', reuse=True):
      A = tf.get_variable('Matrix')
      b = tf.get_variable('Bias')

    # Loss
    beta1 = 0.0
    beta2 = 0.0
    gamma = 0.0
    cost = tf.reduce_mean((Y_hat - Y)**2) + beta1*tf.nn.l2_loss(A) + beta2*tf.nn.l2_loss(C) + gamma*tf.nn.l2_loss(output_diff)
            
    train_op = tf.train.AdamOptimizer(learning_rate=0.001) # add gradient_noise ? 
    #grads_and_vars = train_op.compute_gradients(cost)
    opt_op = train_op.minimize(cost)

  def forward_pass(self):
    """ updates state """
    feed_dict={Y: y_data, U: u_data}
    y_tf, x_tf = sess.run([Y_hat, output], feed_dict=feed_dict)
    return y_tf, x_tf

  def train(self):




