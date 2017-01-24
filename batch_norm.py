from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf


def batch_norm(x,
               n_out,
               is_training,
               reuse=None,
               gamma=None,
               beta=None,
               axes=[0, 1, 2],
               eps=1e-3,
               scope="bn",
               name="bn_out",
               return_mean=False):
  """Applies batch normalization.
    Collect mean and variances on x except the last dimension. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
        x: Input tensor, [B, ...].
        n_out: Integer, depth of input variable.
        gamma: Scaling parameter.
        beta: Bias parameter.
        axes: Axes to collect statistics.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Batch-normalized variable.
        mean: Mean used for normalization (optional).
    """
  with tf.variable_scope(scope, reuse=reuse):
    emean = tf.get_variable("ema_mean", [n_out], trainable=False)
    evar = tf.get_variable("ema_var", [n_out], trainable=False)
    if is_training:
      batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
      batch_mean.set_shape([n_out])
      batch_var.set_shape([n_out])
      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      ema_apply_op_local = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op_local]):
        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
      emean_val = ema.average(batch_mean)
      evar_val = ema.average(batch_var)
      with tf.control_dependencies(
          [tf.assign(emean, emean_val), tf.assign(evar, evar_val)]):
        normed = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, eps, name=name)
    else:
      normed = tf.nn.batch_normalization(
          x, emean, evar, beta, gamma, eps, name=name)
  if return_mean:
    if is_training:
      return normed, mean
    else:
      return normed, emean
  else:
    return normed


def batch_norm_mean_only(x,
                         n_out,
                         is_training,
                         reuse=None,
                         gamma=None,
                         beta=None,
                         axes=[0, 1, 2],
                         scope="bnms",
                         name="bnms_out",
                         return_mean=False):
  """Applies mean only batch normalization.
    Collect mean and variances on x except the last dimension. And apply
    normalization as below:
        x_ = gamma * (x - mean) + beta

    Args:
        x: Input tensor, [B, ...].
        n_out: Integer, depth of input variable.
        gamma: Scaling parameter.
        beta: Bias parameter.
        axes: Axes to collect statistics.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Batch-normalized variable.
        mean: Mean used for normalization (optional).
    """
  with tf.variable_scope(scope, reuse=reuse):
    emean = tf.get_variable("ema_mean", [n_out], trainable=False)
    if is_training:
      batch_mean = tf.reduce_mean(x, axes)
      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      ema_apply_op_local = ema.apply([batch_mean])
      with tf.control_dependencies([ema_apply_op_local]):
        mean = tf.identity(batch_mean)
      emean_val = ema.average(batch_mean)
      with tf.control_dependencies([tf.assign(emean, emean_val)]):
        normed = x - batch_mean
      if gamma is not None:
        normed *= gamma
      if beta is not None:
        normed += beta
    else:
      normed = x - emean
      if gamma is not None:
        normed *= gamma
      if beta is not None:
        normed += beta
  if return_mean:
    if is_training:
      return normed, mean
    else:
      return normed, emean
  else:
    return normed
