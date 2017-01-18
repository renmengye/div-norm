from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf


def l1_loss(x, mean=0.0, alpha=1e-3):
  """Returns L1 regularization loss on the activation.
    l1_loss = alpha * |x - x_mean|

  Args:
    x: Activation tensor.
    mean: Mean tensor of the activation, default=0.0.
    alpha: Regularization constant.

  Returns:
    loss: L1 loss on the mean-centered activation values.
  """
  return alpha * tf.reduce_mean(tf.abs(x - x_mean))
