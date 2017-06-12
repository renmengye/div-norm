from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from utils import logger

log = logger.get()


def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_param=None,
                    wd=None,
                    name=None,
                    trainable=True):
  """Declares a variable.

    Args:
        shape: Shape of the weights, list of int.
        init_method: Initialization method, "constant" or "truncated_normal".
        init_param: Initialization parameters, dictionary.
        wd: Weight decay, float.
        name: Name of the variable, str.
        trainable: Whether the variable can be trained, bool.

    Returns:
        var: Declared variable.
    """
  if dtype != tf.float32:
    log.warning("Not using float32, currently using {}".format(dtype))
  if init_method is None:
    initializer = tf.zeros_initializer(shape, dtype=dtype)
  elif init_method == "truncated_normal":
    if "mean" not in init_param:
      mean = 0.0
    else:
      mean = init_param["mean"]
    if "stddev" not in init_param:
      stddev = 0.1
    else:
      stddev = init_param["stddev"]
    initializer = tf.truncated_normal_initializer(
        mean=mean, stddev=stddev, seed=1, dtype=dtype)
  elif init_method == "uniform_scaling":
    if "factor" not in init_param:
      factor = 1.0
    else:
      factor = init_param["factor"]
    initializer = tf.uniform_unit_scaling_initializer(
        factor=factor, seed=1, dtype=dtype)
  elif init_method == "constant":
    if "val" not in init_param:
      value = 0.0
    else:
      value = init_param["val"]
    initializer = tf.constant_initializer(value=value, dtype=dtype)
  elif init_method == "xavier":
    initializer = tf.contrib.layers.xavier_initializer(
        uniform=False, seed=1, dtype=dtype)
  else:
    raise ValueError("Non supported initialization method!")
  #log.info("Weight shape {}".format(shape))
  if wd is not None:
    if wd > 0.0:
      reg = lambda x: tf.mul(tf.nn.l2_loss(x), wd)
      log.info("Weight decay {}".format(wd))
    else:
      log.warning("No weight decay")
      reg = None
  else:
    log.warning("No weight decay")
    reg = None
  with tf.device("/cpu:0"):
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        regularizer=reg,
        dtype=dtype,
        trainable=trainable)
  return var


def cnn(x,
        filter_size,
        strides,
        pool_fn,
        pool_size,
        pool_strides,
        act_fn,
        dtype=tf.float32,
        add_bias=True,
        wd=None,
        init_std=None,
        init_method=None,
        scope="cnn",
        trainable=True,
        padding="SAME"):
  """Builds a convolutional neural networks.
    Each layer contains the following operations:
        1) Convolution, y = w * x.
        2) Additive bias (optional), y = w * x + b.
        3) Activation function (optional), y = g( w * x + b ).
        4) Pooling (optional).

    Args:
        x: Input variable.
        filter_size: Shape of the convolutional filters, list of 4-d int.
        strides: Convolution strides, list of 4-d int.
        pool_fn: Pooling functions, list of N callable objects.
        pool_size: Pooling field size, list of 4-d int.
        pool_strides: Pooling strides, list of 4-d int.
        act_fn: Activation functions, list of N callable objects.
        add_bias: Whether adding bias or not, bool.
        wd: Weight decay, float.
        scope: Scope of the model, str.
    """
  num_layer = len(filter_size)
  h = x
  with tf.variable_scope(scope):
    for ii in range(num_layer):
      with tf.variable_scope("layer_{}".format(ii)):
        if init_method is not None and init_method[ii]:
          w = weight_variable(
              filter_size[ii],
              init_method=init_method[ii],
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)
        else:
          w = weight_variable(
              filter_size[ii],
              init_method="truncated_normal",
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)

        if add_bias:
          b = weight_variable(
              [filter_size[ii][3]],
              init_method="constant",
              dtype=dtype,
              init_param={"val": 0},
              # wd=wd,       ####### Change this back if it changes anything!!!
              name="b",
              trainable=trainable)
        h = tf.nn.conv2d(
            h, w, strides=strides[ii], padding=padding, name="conv")
        if add_bias:
          h = tf.add(h, b, name="conv_bias")
        if act_fn[ii] is not None:
          h = act_fn[ii](h, name="act")
        if pool_fn[ii] is not None:
          h = pool_fn[ii](h,
                          pool_size[ii],
                          strides=pool_strides[ii],
                          padding="SAME",
                          name="pool")
  return h


def mlp(x,
        dims,
        is_training=True,
        act_fn=None,
        dtype=tf.float32,
        add_bias=True,
        wd=None,
        init_std=None,
        init_method=None,
        scope="mlp",
        dropout=None,
        trainable=True):
  """Builds a multi-layer perceptron.
    Each layer contains the following operations:
        1) Linear transformation, y = w^T x.
        2) Additive bias (optional), y = w^T x + b.
        3) Activation function (optional), y = g( w^T x + b )
        4) Dropout (optional)

    Args:
        x: Input variable.
        dims: Layer dimensions, list of N+1 int.
        act_fn: Activation functions, list of N callable objects.
        add_bias: Whether adding bias or not, bool.
        wd: Weight decay, float.
        scope: Scope of the model, str.
        dropout: Whether to apply dropout, None or list of N bool.
    """
  num_layer = len(dims) - 1
  h = x
  with tf.variable_scope(scope):
    for ii in range(num_layer):
      with tf.variable_scope("layer_{}".format(ii)):
        dim_in = dims[ii]
        dim_out = dims[ii + 1]

        if init_method is not None and init_method[ii]:
          w = weight_variable(
              [dim_in, dim_out],
              init_method=init_method[ii],
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)
        else:
          w = weight_variable(
              [dim_in, dim_out],
              init_method="truncated_normal",
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)

        if add_bias:
          b = weight_variable(
              [dim_out],
              init_method="constant",
              dtype=dtype,
              init_param={"val": 0.0},
              # wd=wd,       ####### Change this back if it changes anything!!!
              name="b",
              trainable=trainable)

        h = tf.matmul(h, w, name="linear")
        if add_bias:
          h = tf.add(h, b, name="linear_bias")
        if act_fn and act_fn[ii] is not None:
          h = act_fn[ii](h)
        if dropout is not None and dropout[ii]:
          log.info("Apply dropout 0.5")
          if is_training:
            keep_prob = 0.5
          else:
            keep_prob = 1.0
          h = tf.nn.dropout(h, keep_prob=keep_prob)
  return h


def concat(x, axis):
  if tf.__version__.startswith("0"):
    return tf.concat(axis, x)
  else:
    return tf.concat(x, axis=axis)


def split(x, num, axis):
  if tf.__version__.startswith("0"):
    return tf.split(axis, num, x)
  else:
    return tf.split(x, num, axis)
