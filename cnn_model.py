from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import nnlib as nn
import numpy as np
import tensorflow as tf
from utils import logger
from l1_reg import l1_loss
from divnorm import div_norm_2d
from layer_norm import layer_norm
from batch_norm import batch_norm, batch_norm_mean_only

log = logger.get()


def get_reg_act(act=tf.nn.relu, l1_reg=0.0, l1_collection=None):
  """Gets a regular activation function, with L1 regularization.

  Args:
    act: Activation function.
    l1_reg: L1 regularization constant, on the centered pre-activation.
  """

  def _act(x, name="act"):
    """Regular activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    if l1_reg > 0.0:
      l1_collection.append(l1_loss(x, alpha=l1_reg))
    return act(x, name=name)

  return _act


def get_bn_act(act=tf.nn.relu,
               is_training=True,
               affine=True,
               sigma_init=1e-2,
               l1_reg=0.0,
               scope="bn",
               mask=None,
               axes=[0, 1, 2],
               l1_collection=None,
               learn_sigma=False,
               dtype=tf.float32):
  """Gets a batch-normalized activation function, with L1 regularization.

  Args:
      is_training: Whether in training mode (use mini-batch statistics or
      full batch statistics), bool.
      act: Activation function, callable object.
      affine: Whether to add affine transformation, bool.
      eps: Denominator bias, float.
      l1_reg: L1 regularization constant, on the centered pre-activation.
      scope: Scope of the operation, str.
  """

  def _act(x, name="bn_act"):
    """Batch-normalized activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    n_out = x.get_shape()[-1]
    with tf.variable_scope("bn_params"):
      if affine:
        beta = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
      if learn_sigma:
        sigma = nn.weight_variable(
            [1],
            init_method="constant",
            dtype=dtype,
            init_param={"val": sigma_init},
            name="sigma")
      else:
        sigma = sigma_init
      eps = sigma**2
    x_normed, x_mean = batch_norm(
        x,
        n_out,
        is_training,
        gamma=gamma,
        beta=beta,
        eps=eps,
        axes=axes,
        scope=scope,
        name=name,
        return_mean=True)
    if l1_reg > 0.0:
      l1_collection.append(l1_loss(x, x_mean=x_mean, alpha=l1_reg))
    return act(x_normed)

  if mask is None or mask == True:
    return _act
  else:
    log.warning("Not using batch norm.")
    return get_reg_act(act, 0.0)


def get_bnms_act(act=tf.nn.relu,
                 is_training=True,
                 affine=True,
                 l1_reg=0.0,
                 scope="bn",
                 mask=None,
                 axes=[0, 1, 2],
                 l1_collection=None,
                 dtype=tf.float32):
  """Gets a mean subtracted version of batch-normalized activation function, with L1 regularization.

  Args:
      is_training: Whether in training mode (use mini-batch statistics or
      full batch statistics), bool.
      act: Activation function, callable object.
      affine: Whether to add affine transformation, bool.
      eps: Denominator bias, float.
      l1_reg: L1 regularization constant, on the centered pre-activation.
      scope: Scope of the operation, str.
  """

  def _act(x, name="bn_act"):
    """Batch-normalized activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    n_out = x.get_shape()[-1]
    with tf.variable_scope("bn_params"):
      if affine:
        beta = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
    x_normed, x_mean = batch_norm_mean_only(
        x,
        n_out,
        is_training,
        gamma=gamma,
        beta=beta,
        axes=axes,
        scope=scope,
        name=name,
        return_mean=True)
    if l1_reg > 0.0:
      l1_collection.append(l1_loss(x, x_mean=x_mean, alpha=l1_reg))
    return act(x_normed)

  if mask is None or mask == True:
    return _act
  else:
    log.warning("Not using batch norm.")
    return get_reg_act(act, 0.0)


def get_ln_act(act=tf.nn.relu,
               affine=True,
               sigma_init=1e-2,
               l1_reg=0.0,
               scope="ln",
               axes=[1, 2, 3],
               l1_collection=None,
               learn_sigma=False,
               dtype=tf.float32):
  """Gets a layer-normalized activation function.

  Args:
      act: Activation function, callable object.
      affine: Whether to add affine transformation, bool.
      eps: Denominator bias, float.
      l1_reg: L1 regularization constant, on the centered pre-activation.
      scope: Scope of the operation, str.
  """

  def _act(x, name="ln_act"):
    """Layer-normalized activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    n_out = x.get_shape()[-1]
    with tf.variable_scope(scope):
      if affine:
        beta = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
      if learn_sigma:
        sigma = nn.weight_variable(
            [1],
            init_method="constant",
            dtype=dtype,
            init_param={"val": sigma_init},
            name="sigma")
      else:
        sigma = sigma_init
      eps = sigma**2
    x_normed, x_mean = layer_norm(
        x,
        gamma=gamma,
        beta=beta,
        axes=axes,
        eps=eps,
        name=name,
        return_mean=True)
    if l1_reg > 0.0:
      l1_collection.append(l1_loss(x, x_mean=x_mean, alpha=l1_reg))
    return act(x_normed)

  return _act


def get_lnms_act(act=tf.nn.relu,
                 affine=True,
                 l1_reg=0.0,
                 scope="ln",
                 axes=[1, 2, 3],
                 l1_collection=None,
                 dtype=tf.float32):
  """Gets a layer-normalized activation function.

  Args:
      act: Activation function, callable object.
      affine: Whether to add affine transformation, bool.
      eps: Denominator bias, float.
      l1_reg: L1 regularization constant, on the centered pre-activation.
      scope: Scope of the operation, str.
  """

  def _act(x, name="ln_act"):
    """Layer-normalized activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    n_out = x.get_shape()[-1]
    with tf.variable_scope(scope):
      if affine:
        beta = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
    x_mean = tf.reduce_mean(x, axes, keep_dims=True)
    x_normed = x - x_mean
    if gamma is not None:
      x_normed *= gamma
    if beta is not None:
      x_normed += beta
    if l1_reg > 0.0:
      l1_collection.append(l1_loss(x, x_mean=x_mean, alpha=l1_reg))
    return act(x_normed)

  return _act


def get_dn_act(sum_window=[5, 5],
               sup_window=[5, 5],
               act=tf.nn.relu,
               affine=False,
               sigma_init=1e-2,
               l1_reg=0.0,
               scope="dn",
               l1_collection=None,
               learn_sigma=False,
               dtype=tf.float32):
  """Gets a divisive-normalized activation function.

  Args:
      sum_window: Summation window size, [H_sum, W_sum].
      sup_winow: Suppression window size, [H_sup, W_sup].
      act: Activation function, callable object.
      affine: Whether to add affine transformation, bool.
      eps: Denominator bias, float.
      l1_reg: L1 regularization constant, on the centered pre-activation.
      scope: Scope of the operation, str.
  """

  def _act(x, name="dn_act"):
    """Divisive-normalized activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    n_out = x.get_shape()[-1]
    with tf.variable_scope(scope):
      if affine:
        beta = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
      if learn_sigma:
        sigma = nn.weight_variable(
            [1],
            init_method="constant",
            dtype=dtype,
            init_param={"val": sigma_init},
            name="sigma")
      else:
        sigma = sigma_init
      eps = sigma**2
    x_normed, x_mean = div_norm_2d(
        x,
        sum_window,
        sup_window,
        gamma=gamma,
        beta=beta,
        eps=eps,
        name=name,
        return_mean=True)
    if l1_reg > 0.0:
      l1_collection.append(l1_loss(x, x_mean=x_mean, alpha=l1_reg))
    return act(x_normed)

  if sum_window is not None and sup_window is not None:
    return _act
  else:
    log.warning("Not using divisive norm.")
    return get_reg_act(act=act, l1_reg=0.0)


def get_dnms_act(sum_window=[5, 5],
                 act=tf.nn.relu,
                 affine=False,
                 l1_reg=0.0,
                 scope="dnms",
                 l1_collection=None,
                 dtype=tf.float32):
  """Gets a mean-subtracted version of divisive-normalized activation function.

  Args:
      sum_window: Summation window size, [H_sum, W_sum].
      sup_winow: Suppression window size, [H_sup, W_sup].
      act: Activation function, callable object.
      affine: Whether to add affine transformation, bool.
      l1_reg: L1 regularization constant, on the centered pre-activation.
      scope: Scope of the operation, str.
  """

  def _act(x, name="dnms_act"):
    """Divisive-normalized activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    n_out = x.get_shape()[-1]
    with tf.variable_scope(scope):
      if affine:
        beta = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            [n_out],
            init_method="constant",
            dtype=dtype,
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
      w_sum = tf.ones(sum_window + [1, 1]) / np.prod(np.array(sum_window))
      x_mean = tf.reduce_mean(x, [3], keep_dims=True)
      x_mean = tf.nn.conv2d(x_mean, w_sum, strides=[1, 1, 1, 1], padding='SAME')
      x_normed = x - x_mean
      if gamma is not None:
        x_normed *= gamma
      if beta is not None:
        x_normed *= beta
      if l1_reg > 0.0:
        l1_collection.append(l1_loss(x, x_mean, alpha=l1_reg))
      return act(x_normed)

  if sum_window is not None:
    return _act
  else:
    log.warning("Not using divisive norm.")
    return get_reg_act(act=act, l1_reg=0.0)


def get_tf_fn(name):
  """Gets a tensorflow function."""
  if name == "avg_pool":
    return tf.nn.avg_pool
  elif name == "max_pool":
    return tf.nn.max_pool
  elif name == "relu":
    return tf.nn.relu
  elif name is None:
    return None
  else:
    raise Exception("Unknown tf function \"{}\"".format(name))


class CNNModel(object):
  """CNN model."""

  def __init__(self,
               config,
               is_training=True,
               inference_only=False,
               inp=None,
               label=None):
    """Builds CNN model.

    Args:
        config: Config object.
        is_training: bool.
    """
    self._config = config
    self._is_training = is_training
    self._l1_collection = []

    self._num_cnn_layer = len(config.filter_size)
    self._num_mlp_layer = len(config.mlp_dims) - 1

    # Input.
    if inp is None:
      x = tf.placeholder(
          self.dtype(), [None, config.height, config.width, config.num_channel])
    else:
      x = inp
    if label is None:
      y = tf.placeholder(tf.int32, [None])
    else:
      y = label

    logits = self.build_inference_network(x)
    predictions = tf.nn.softmax(logits)

    # Compute cross-entropy loss.
    cross_ent = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)) / tf.cast(
            tf.shape(x)[0], dtype=self.dtype())
    total_loss = cross_ent

    self._input = x
    self._label = y
    self._output = predictions

    # Regularization
    total_loss += self._decay()
    total_loss += self._l1_loss()
    self._cross_ent = cross_ent
    self._cost = total_loss

    if not is_training or inference_only:
      return

    # Optimizer.
    global_step = tf.Variable(
        0.0, name="global_step", dtype=self.dtype(), trainable=False)
    lr = tf.Variable(
        0.0, name="learn_rate", dtype=self.dtype(), trainable=False)
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config.momentum)
    train_step = opt.minimize(total_loss, global_step=global_step)

    self._new_lr = tf.placeholder(
        self.dtype(), shape=[], name="new_learning_rate")
    self._lr = lr
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self._train_op = train_step
    self._global_step = global_step

  def dtype(self):
    tensor_type = os.getenv("TF_DTYPE", "float32")
    if tensor_type == "float32":
      return tf.float32
    elif tensor_type == "float64":
      return tf.float64
    else:
      raise Exception("Unknown tensor type {}".format(tensor_type))

  def build_inference_network(self, x):
    """Build inference part of the network."""
    config = self.config
    is_training = self.is_training

    # Activation functions (combining normalization).
    if config.norm_field == "batch":
      log.info("Using batch normalization")
      log.info("Setting sigma={:.3e}".format(config.sigma_init))
      log.info("Setting sigma learnable={}".format(config.learn_sigma))
      log.info("Setting L1={:.3e}".format(config.l1_reg))
      conv_act_fn = [
          get_bn_act(
              act=get_tf_fn(aa),
              is_training=is_training,
              sigma_init=config.sigma_init,
              affine=config.norm_affine,
              l1_reg=config.l1_reg,
              mask=config.bn_mask[ii],
              l1_collection=self.l1_collection,
              learn_sigma=config.learn_sigma,
              dtype=self.dtype()) for ii, aa in enumerate(config.conv_act_fn)
      ]
    elif config.norm_field == "batch_ms":
      log.info("Using mean subtracted batch normalization")
      log.info("Setting L1={:.3e}".format(config.l1_reg))
      conv_act_fn = [
          get_bnms_act(
              act=get_tf_fn(aa),
              is_training=is_training,
              affine=config.norm_affine,
              l1_reg=config.l1_reg,
              mask=config.bn_mask[ii],
              l1_collection=self.l1_collection,
              dtype=self.dtype()) for ii, aa in enumerate(config.conv_act_fn)
      ]
    elif config.norm_field == "layer":
      log.info("Using layer normalization")
      log.info("Setting sigma={:.3e}".format(config.sigma_init))
      log.info("Setting sigma learnable={}".format(config.learn_sigma))
      log.info("Setting L1={:.3e}".format(config.l1_reg))
      conv_act_fn = [
          get_ln_act(
              act=get_tf_fn(aa),
              sigma_init=config.sigma_init,
              affine=config.norm_affine,
              l1_reg=config.l1_reg,
              l1_collection=self.l1_collection,
              learn_sigma=config.learn_sigma,
              dtype=self.dtype()) for ii, aa in enumerate(config.conv_act_fn)
      ]
    elif config.norm_field == "layer_ms":
      log.info("Using mean subtracted layer normalization")
      log.info("Setting L1={:.3e}".format(config.l1_reg))
      conv_act_fn = [
          get_lnms_act(
              act=get_tf_fn(aa),
              affine=config.norm_affine,
              l1_reg=config.l1_reg,
              l1_collection=self.l1_collection,
              dtype=self.dtype()) for ii, aa in enumerate(config.conv_act_fn)
      ]
    elif config.norm_field == "div":
      log.info("Using divisive normalization")
      log.info("Setting sigma={:.3e}".format(config.sigma_init))
      log.info("Setting sigma learnable={}".format(config.learn_sigma))
      log.info("Setting L1={:.3e}".format(config.l1_reg))
      conv_act_fn = [
          get_dn_act(
              sum_window=config.sum_window[ii],
              sup_window=config.sup_window[ii],
              act=get_tf_fn(aa),
              sigma_init=config.sigma_init,
              affine=config.norm_affine,
              l1_reg=config.l1_reg,
              l1_collection=self.l1_collection,
              learn_sigma=config.learn_sigma,
              dtype=self.dtype()) for ii, aa in enumerate(config.conv_act_fn)
      ]
    elif config.norm_field == "div_ms":
      log.info("Using mean subtracted divisive normalization")
      log.info("Setting L1={:.3e}".format(config.l1_reg))
      conv_act_fn = [
          get_dnms_act(
              sum_window=config.sum_window[ii],
              act=get_tf_fn(aa),
              affine=config.norm_affine,
              l1_reg=config.l1_reg,
              l1_collection=self.l1_collection,
              dtype=self.dtype()) for ii, aa in enumerate(config.conv_act_fn)
      ]
    elif config.norm_field == "no" or config.norm_field is None:
      log.info("Not using normalization")
      log.info("Setting L1={:.3e}".format(config.l1_reg))
      conv_act_fn = [
          get_reg_act(
              get_tf_fn(aa),
              l1_reg=config.l1_reg,
              l1_collection=self.l1_collection) for aa in config.conv_act_fn
      ]
    else:
      raise Exception("Unknown normalization \"{}\"".format(config.norm_field))

    # Pooling functions.
    pool_fn = [get_tf_fn(pp) for pp in config.pool_fn]

    # CNN function.
    cnn_fn = lambda x: nn.cnn(x, config.filter_size,
                              strides=config.strides,
                              pool_fn=pool_fn,
                              pool_size=config.pool_size,
                              pool_strides=config.pool_strides,
                              act_fn=conv_act_fn,
                              dtype=self.dtype(),
                              add_bias=True,
                              init_std=config.conv_init_std,
                              init_method=config.conv_init_method,
                              wd=config.wd)

    # MLP function.
    mlp_act_fn = [get_tf_fn(aa) for aa in config.mlp_act_fn]
    mlp_fn = lambda x: nn.mlp(x, config.mlp_dims,
                              is_training=is_training,
                              act_fn=mlp_act_fn,
                              dtype=self.dtype(),
                              init_std=config.mlp_init_std,
                              init_method=config.mlp_init_method,
                              dropout=config.mlp_dropout,
                              wd=config.wd)

    # Prediction model.
    h = cnn_fn(x)
    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    h = tf.reshape(h, [-1, config.mlp_dims[0]])
    logits = mlp_fn(h)
    return logits

  def assign_weight(self, weights):
    all_variables = tf.global_variables()

    var_dict = {}
    for var in all_variables:
      var_dict[var.name] = var

    assign_ops = []
    for ii in xrange(self._num_cnn_layer):
      var_name = 'Model/cnn/layer_{:d}/w:0'.format(ii)
      key_name = 'w_{:d}'.format(ii)
      assign_ops += [tf.assign(var_dict[var_name], weights[key_name])]

      var_name = 'Model/cnn/layer_{:d}/b:0'.format(ii)
      key_name = 'b_{:d}'.format(ii)
      assign_ops += [tf.assign(var_dict[var_name], weights[key_name])]

    for ii in xrange(self._num_cnn_layer,
                     self._num_cnn_layer + self._num_mlp_layer):
      var_name = 'Model/mlp/layer_{:d}/w:0'.format(ii - self._num_cnn_layer)
      key_name = 'w_{:d}'.format(ii)
      assign_ops += [tf.assign(var_dict[var_name], weights[key_name])]

      var_name = 'Model/mlp/layer_{:d}/b:0'.format(ii - self._num_cnn_layer)
      key_name = 'b_{:d}'.format(ii)
      assign_ops += [tf.assign(var_dict[var_name], weights[key_name])]

    return tf.group(*assign_ops)

  def _decay(self):
    """L2 weight decay loss."""
    wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    log.info("Weight decay variables: {}".format(wd_losses))
    log.info("Total length: {}".format(len(wd_losses)))
    if len(wd_losses) > 0:
      return tf.add_n(wd_losses)
    else:
      log.warning("No weight decay variables!")
      return 0.0

  def _l1_loss(self):
    """L1 activation loss."""
    # l1_reg_losses = tf.get_collection(L1_REG_KEY)
    if len(self.l1_collection) > 0:
      log.warning("L1 Regularizers {}".format(self.l1_collection))
      return tf.add_n(self.l1_collection)
    else:
      log.warning("No L1 loss variables!")
      return 0.0

  @property
  def input(self):
    return self._input

  @property
  def output(self):
    return self._output

  @property
  def label(self):
    return self._label

  @property
  def config(self):
    return self._config

  @property
  def is_training(self):
    return self._is_training

  @property
  def cost(self):
    return self._cost

  @property
  def cross_ent(self):
    return self._cross_ent

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def global_step(self):
    return self._global_step

  @property
  def l1_collection(self):
    return self._l1_collection

  def assign_lr(self, session, lr_value):
    """Assigns new learning rate."""
    log.info("Adjusting learning rate to {}".format(lr_value))
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def infer_step(self, sess, inp):
    """Run inference."""
    return sess.run(self.output, feed_data={self.input: inp})

  def train_step(self, sess, inp, label):
    """Run training."""
    feed_data = {self.input: inp, self.label: label}
    cost, ce, _ = sess.run([self.cost, self.cross_ent, self.train_op],
                           feed_dict=feed_data)
    return ce
