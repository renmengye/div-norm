# Author: Mengye Ren (mren@cs.toronto.edu).
#
# Modified from Tensorflow original code.
# Original Tensorflow license shown below.
# =============================================================================
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import nnlib as nn
import numpy as np
import tensorflow as tf

from utils import logger
from cnn_model import get_reg_act, get_bn_act, get_ln_act, get_dn_act

log = logger.get()


class ResNetModel(object):
  """ResNet model."""

  def __init__(self,
               config,
               is_training=True,
               inference_only=False,
               inp=None,
               label=None):
    """ResNet constructor.

    Args:
      config: Hyperparameters.
      is_training: One of "train" and "eval".
      inference_only: Do not build optimizer.
    """
    self._config = config
    self._l1_collection = []
    self.is_training = is_training

    # Input.
    if inp is None:
      x = tf.placeholder(
          tf.float32, [None, config.height, config.width, config.num_channel])
    else:
      x = inp

    if label is None:
      y = tf.placeholder(tf.int32, [None])
    else:
      y = label

    logits = self.build_inference_network(x)
    predictions = tf.nn.softmax(logits)

    with tf.variable_scope("costs"):
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
      xent = tf.reduce_sum(xent, name="xent") / tf.to_float(tf.shape(x)[0])
      cost = xent
      cost += self._decay()
      cost += self._l1_loss()

    self._cost = cost
    self._input = x
    self._label = y
    self._cross_ent = xent
    self._output = predictions

    if not is_training or inference_only:
      return

    global_step = tf.Variable(0.0, name="global_step", trainable=False)
    lr = tf.Variable(0.0, name="learn_rate", trainable=False)
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(cost, trainable_variables)
    if config.optimizer == "sgd":
      optimizer = tf.train.GradientDescentOptimizer(lr)
    elif config.optimizer == "mom":
      optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    train_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=global_step,
        name="train_step")
    self._train_op = train_op
    self._global_step = global_step
    self._lr = lr
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    """Assigns new learning rate."""
    log.info("Adjusting learning rate to {}".format(lr_value))
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def cost(self):
    return self._cost

  @property
  def train_op(self):
    return self._train_op

  @property
  def config(self):
    return self._config

  @property
  def lr(self):
    return self._lr

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
  def cross_ent(self):
    return self._cross_ent

  @property
  def global_step(self):
    return self._global_step

  @property
  def l1_collection(self):
    return self._l1_collection

  def build_inference_network(self, x):
    config = self.config
    is_training = self.is_training
    num_stages = len(self.config.num_residual_units)

    if config.norm_field is None:
      act_fn = [None] * (num_stages + 1)
      # raise Exception("No normalization schemes specified.")
    elif config.norm_field == "no":
      log.warning("No normalization scheme selected!")
      act_fn = [
          get_reg_act(
              act=tf.nn.relu,
              l1_reg=config.l1_reg,
              l1_collection=self.l1_collection)
      ] * (num_stages + 1)
    elif config.norm_field == "batch":
      act_fn = [
          get_bn_act(
              act=tf.nn.relu,
              is_training=is_training,
              l1_reg=config.l1_reg,
              sigma_init=config.sigma_init,
              affine=config.norm_affine,
              l1_collection=self.l1_collection,
              learn_sigma=config.learn_sigma)
      ] * (num_stages + 1)
    elif config.norm_field == "layer":
      act_fn = [
          get_ln_act(
              act=tf.nn.relu,
              l1_reg=config.l1_reg,
              sigma_init=config.sigma_init**2,
              affine=config.norm_affine,
              l1_collection=self.l1_collection,
              learn_sigma=config.learn_sigma)
      ] * (num_stages + 1)
    elif config.norm_field == "div":
      act_fn = [
          get_dn_act(
              sum_window=config.sum_window[ii],
              sup_window=config.sup_window[ii],
              act=tf.nn.relu,
              l1_reg=config.l1_reg,
              sigma_init=config.sigma_init,
              affine=config.norm_affine,
              l1_collection=self.l1_collection,
              learn_sigma=config.learn_sigma) for ii in range((num_stages + 1))
      ]

    # Regular activation function here for hack.
    reg_act_fn = get_reg_act(
        act=tf.nn.relu, l1_reg=0.0, l1_collection=self.l1_collection)

    strides = config.strides
    activate_before_residual = config.activate_before_residual
    filters = config.filters
    init_filter = config.init_filter

    with tf.variable_scope("init"):
      h = self._conv("init_conv", x, init_filter, init_filter, filters[0],
                     self._stride_arr(config.init_stride))

      # Max-pooling is used in ImageNet experiments to further reduce
      # dimensionality.
      if config.init_max_pool:
        h = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

    if config.use_bottleneck:
      res_func = self._bottleneck_residual
      # For CIFAR-10 it's [16, 16, 32, 64] => [16, 64, 128, 256]
      for ii in range(1, len(filters)):
        filters[ii] *= 4
    else:
      res_func = self._residual

    for ss in range(num_stages):
      with tf.variable_scope("unit_{}_0".format(ss + 1)):
        h = res_func(
            h,
            filters[ss],
            filters[ss + 1],
            self._stride_arr(strides[ss]),
            act_fn=act_fn[ss],
            activate_before_residual=activate_before_residual[ss])
      for ii in range(1, config.num_residual_units[ss]):
        with tf.variable_scope("unit_{}_{}".format(ss + 1, ii)):
          if config.stagewise_norm:
            _act_fn = reg_act_fn
          else:
            _act_fn = act_fn[ss]
          h = res_func(
              h,
              filters[ss + 1],
              filters[ss + 1],
              self._stride_arr(1),
              act_fn=_act_fn,
              activate_before_residual=False)

    with tf.variable_scope("unit_last"):
      if act_fn[num_stages] is None:
        h = self._batch_norm("final_bn", h)
        h = self._relu(h, config.relu_leakiness)
      else:
        h = act_fn[num_stages](h)
      h = self._global_avg_pool(h)

    with tf.variable_scope("logit"):
      logits = self._fully_connected(h, config.num_classes)

    return logits

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      n_out = x.get_shape()[-1]
      beta = nn.weight_variable(
          [n_out], init_method="constant", init_param={"val": 0.0}, name="beta")
      gamma = nn.weight_variable(
          [n_out],
          init_method="constant",
          init_param={"val": 1.0},
          name="gamma")
      return nn.batch_norm(
          x,
          n_out,
          self.is_training,
          reuse=None,
          gamma=gamma,
          beta=beta,
          axes=[0, 1, 2],
          eps=1e-3,
          scope="bn",
          name="bn_out",
          return_mean=False)

  def _residual(self,
                x,
                in_filter,
                out_filter,
                stride,
                act_fn=None,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope("shared_activation"):
        if act_fn is None:
          x = self._batch_norm("init_bn", x)
          x = self._relu(x, self.config.relu_leakiness)
        else:
          x = act_fn(x)
        orig_x = x
    else:
      with tf.variable_scope("residual_only_activation"):
        orig_x = x
        if act_fn is None:
          x = self._batch_norm("init_bn", x)
          x = self._relu(x, self.config.relu_leakiness)
        else:
          x = act_fn(x)

    with tf.variable_scope("sub1"):
      x = self._conv("conv1", x, 3, in_filter, out_filter, stride)

    with tf.variable_scope("sub2"):
      if act_fn is None:
        x = self._batch_norm("bn2", x)
        x = self._relu(x, self.config.relu_leakiness)
      else:
        x = act_fn(x)
      x = self._conv("conv2", x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope("sub_add"):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, "VALID")
        orig_x = tf.pad(
            orig_x,
            [[0, 0], [0, 0], [0, 0],
             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
      x += orig_x
    log.info("image after unit {}".format(x.get_shape()))
    return x

  def _bottleneck_residual(self,
                           x,
                           in_filter,
                           out_filter,
                           stride,
                           act_fn=None,
                           activate_before_residual=False):
    """Bottleneck resisual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope("common_bn_relu"):
        if act_fn is None:
          x = self._batch_norm("init_bn", x)
          x = self._relu(x, self.config.relu_leakiness)
        else:
          x = act_fn(x)
        orig_x = x
    else:
      with tf.variable_scope("residual_bn_relu"):
        orig_x = x
        if act_fn is None:
          x = self._batch_norm("init_bn", x)
          x = self._relu(x, self.config.relu_leakiness)
        else:
          x = act_fn(x)

    with tf.variable_scope("sub1"):
      x = self._conv("conv1", x, 1, in_filter, out_filter / 4, stride)

    with tf.variable_scope("sub2"):
      if act_fn is None:
        x = self._batch_norm("bn2", x)
        x = self._relu(x, self.config.relu_leakiness)
      else:
        x = act_fn(x)
      x = self._conv("conv2", x, 3, out_filter / 4, out_filter / 4,
                     [1, 1, 1, 1])

    with tf.variable_scope("sub3"):
      if act_fn is None:
        x = self._batch_norm("bn3", x)
        x = self._relu(x, self.config.relu_leakiness)
      else:
        x = act_fn(x)
      x = self._conv("conv3", x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope("sub_add"):
      if in_filter != out_filter:
        orig_x = self._conv("project", orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    log.info("image after unit {}".format(x.get_shape()))
    return x

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
    if len(self.l1_collection) > 0:
      log.warning("L1 Regularizers {}".format(self.l1_collection))
      return tf.add_n(self.l1_collection)
    else:
      log.warning("No L1 loss variables!")
      return 0.0

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = nn.weight_variable(
          [filter_size, filter_size, in_filters, out_filters],
          init_method="truncated_normal",
          init_param={"mean": 0,
                      "stddev": np.sqrt(2.0 / n)},
          wd=self.config.wd,
          name="w")
      return tf.nn.conv2d(x, kernel, strides, padding="SAME")

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x_shape = x.get_shape()
    d = x_shape[1]
    w = nn.weight_variable(
        [d, out_dim],
        init_method="uniform_scaling",
        init_param={"factor": 1.0},
        wd=self.config.wd,
        name="w")
    b = nn.weight_variable(
        [out_dim], init_method="constant", init_param={"val": 0.0}, name="b")
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
