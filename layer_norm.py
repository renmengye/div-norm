from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf


def layer_norm(x,
               gamma=None,
               beta=None,
               axes=[1, 2, 3],
               eps=1e-3,
               scope="ln",
               name="ln_out",
               return_mean=False):
  """Applies layer normalization.
    Collect mean and variances on x except the first dimension. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)

    Args:
        x: Input tensor, [B, ...].
        axes: Axes to collect statistics.
        gamma: Scaling parameter.
        beta: Bias parameter.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Layer-normalized variable.
        mean: Mean used for normalization (optional).
    """
  with tf.variable_scope(scope):
    x_shape = [x.get_shape()[-1]]
    mean, var = tf.nn.moments(x, axes, name='moments', keep_dims=True)
    normed = (x - mean) / tf.sqrt(eps + var)
    if gamma is not None:
      normed *= gamma
    if beta is not None:
      normed += beta
    normed = tf.identity(normed, name=name)
  if return_mean:
    return normed, mean
  else:
    return normed
