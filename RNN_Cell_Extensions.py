from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import GRUCell
from tensorflow.compat.v1.nn.rnn_cell import LSTMStateTuple
import _threading_local


class LinearSpaceDecoderWrapper(GRUCell):
  """Operator adding a linear encoder to an RNN cell"""

  def __init__(self, cell, output_size,**kwargs):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    #super(LinearSpaceDecoderWrapper,self).__init__(cell**kwargs)
    if not isinstance(cell, GRUCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    super(LinearSpaceDecoderWrapper,self).__init__(units = cell.state_size,**kwargs)

    print( 'output_size = {0}'.format(output_size) )
    print( 'state_size = {0}'.format(cell.state_size) )

    # Tuple if multi-rnn
    if isinstance(cell.state_size,tuple):

      # Fine if GRU...
      insize = cell.state_size[-1]

      # LSTMStateTuple if LSTM
      if isinstance( insize, LSTMStateTuple ):
        insize = insize.h

    else:
      # Fine if not multi-rnn
      insize = cell.state_size

    #self._thread_local = []
    """
    self.w_out = tf.compat.v1.get_variable("proj_w_out",
        [insize, output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
    self.b_out = tf.compat.v1.get_variable("proj_b_out", [output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
    """
    self.cell = cell
    self.input_size = cell.state_size
    self.linear_output_size = output_size

  """
  @property
  def state_size(self):
    return self._cell.state_size
  
  @property
  def output_size(self):
    return self.linear_output_size
  """

  def build(self, input_shape):
    self.w = self.add_weight(name="W",
      shape=(self.input_size, self.linear_output_size),
      initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04),
      trainable=True,
    )
    self.b = self.add_weight(name='b',
      shape=(self.linear_output_size,), initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04), trainable=True
    )

  def call(self, inputs, state, scope=None):
    """Use a linear layer and pass the output to the cell."""

    # Run the rnn as usual
    output, new_state = self.cell(inputs, state, scope)

    # Apply the multiplication to everything
    output = tf.matmul(output, self.w) + self.b

    return output, new_state