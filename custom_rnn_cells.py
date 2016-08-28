"""
  Like rnn_cells.py from tensorflow, but with custom defined cells
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
# Comment out nest import for Yeti cluster (why?)
#from tensorflow.python.util import nest

### custom
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell
## /end custom

class BasicRNNCellNoise(rnn_cell.RNNCell):
  """The most basic RNN cell with noise"""

  def __init__(self, num_units, input_size=None, activation=tanh, stddev=0.0):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._stddev = stddev

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B + noise)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = self._activation(rnn_cell._linear([inputs, state + random_ops.random_normal([self._num_units], stddev=self._stddev)], self._num_units, True))
    return output, output
