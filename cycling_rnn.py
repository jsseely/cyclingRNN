""" Cycling RNN """

import numpy as np
import tensorflow as tf

class RNN(object):
    def __init__(self, input_dim, state_dim,
                output_dim, num_sequences, seq_length):
        """
        """
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.num_sequences = num_sequences
        self.seq_length = seq_length

        X = []
        for s in range(self.num_sequences):
            X.append(tf.Variable(tf.random_normal([])))
        X = tf.Variable(tf.random_normal())
        x0 = tf.Variable(tf.random_normal([total_batches,state_dim], stddev=0.005), name='x0')

        C = tf.Variable(tf.random_normal([n,p], stddev=0.5), name='C')
        d = tf.Variable(tf.constant(0.1, shape=[1,p]), name='d')

        U = tf.placeholder(tf.float32, [time_steps, None, n], name='U')
        Y = tf.placeholder(tf.float32, [time_steps, None, p], name='Y')
        Inds = tf.placeholder(tf.int32, [None,], name='batch_inds')

        cell = tf.nn.rnn_cell.BasicRNNCell(n)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        output, _ = tf.nn.dynamic_rnn(cell, U, initial_state=tf.gather(x0, Inds), dtype=tf.float32, time_major=True)

        Y_hat = tf.unpack(output)
        Y_hat = [tf.matmul(Y_hat[i], C) + d for i in range(time_steps)]
        Y_hat = tf.pack(Y_hat)

        output_diff = tf.gather(output, range(1,time_steps))-tf.gather(output, range(0,time_steps-1))

        Y_fetch = tf.identity(Y)

        #Get A matrix
        with tf.variable_scope('RNN/BasicRNNCell/Linear', reuse=True):
            A = tf.get_variable('Matrix')
