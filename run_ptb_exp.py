#!/usr/bin/python
"""
Author: Mengye Ren (mren@cs.toronto.edu)
Modified from Tensorflow Library.

The following code explores different normalization schemes in RNN on
PennTree bank dataset.

Usage:
python run_ptb_exp.py    --model           [MODEL NAME]        \
                         --config          [CONFIG FILE]       \
                         --env             [ENV FILE]          \
                         --data_folder     [DATASET FOLDER]    \
                         --logs            [LOGS FOLDER]       \
                         --results         [SAVE FOLDER]       \
                         --gpu             [GPU ID]

Flags:
    --model: Model type. Available options are:
        1) base-[rnn]
        2) bn-[rnn]
        3) bn-l1-[rnn]
        4) ln-[rnn]
        5) ln-s-[rnn]
        6) ln-l1-[rnn]
        7) ln-star-[rnn]
        8) dn-[rnn]
        9) dn-star-[rnn]
        where [rnn] can be one of "lstm", "tanh-rnn", and "relu-rnn"
    --config: Not using the pre-defined configs above, specify the JSON file
    that contains model configurations.
    --data_folder: Path to data folder, default is ../data/ptb.
    --logs: Path to logs folder, default is ../logs/default.
    --results: Path to save folder, default is ../results.
    --gpu: Which GPU to run, default is 0, -1 for CPU.
"""

# =============================================================================
# Original Tensorflow License is below.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import os
import time
import numpy as np
import sys
import tensorflow as tf

from data import ptb_reader as reader
from expr import BasicEnvironment, gen_id, get_device
from utils import logger
from utils import progress_bar as pb
from batch_norm import batch_norm
from layer_norm import layer_norm
from div_norm import div_norm_1d

import nnlib as nn
import ptb_exp_config as conf

_logs_folder = "logs/default"
_save_folder = "results/ptb"
_data_folder = "ptb"

flags = tf.flags
flags.DEFINE_string("config", None, "manually defined config file")
flags.DEFINE_string("data_folder", _data_folder, "Where data is stored.")
flags.DEFINE_string("description", None, "Experiment description.")
flags.DEFINE_string("env", None, "manually set environ file")
flags.DEFINE_integer("gpu", 0, "GPU ID")
flags.DEFINE_string("id", None, "experiment ID")
flags.DEFINE_string("logs", _logs_folder, "Logs folder.")
flags.DEFINE_string("model", "base-lstm",
                    "Model type: small, medium, or large.")
flags.DEFINE_string("results", _save_folder, "Model output directory.")
flags.DEFINE_bool("verbose", False, "verbose logging")

FLAGS = flags.FLAGS

log = logger.get()

L1_REG_KEY = "L1_REG"


def data_type():
  # return tf.float16 if FLAGS.use_fp16 else tf.float32
  return tf.float32


def get_reg_act(act, l1_reg=0.0, scope="ln_act"):
  """Gets a regular activation function, with L1 regularization.

  Args:
    act: Activation function.
    l1_reg: L1 regularization constant.
  """

  def _act(x, reuse=None, name="act_out"):
    """Regular activation function.

    Args:
        x: Input tensor.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    with tf.variable_scope(scope):
      if l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY, l1_reg * tf.reduce_mean(tf.abs(x)))
      return act(x, name=name)

  return _act


def get_ln_act(act, affine=True, eps=1e-3, l1_reg=0.0, scope="ln_act"):
  """Gets a layer-normalized activation function.

  Args:
      act: Activation function, callable object.
      affine: Whether to add affine transformation, bool.
      eps: Denominator bias, float.
      scope: Scope of the operation, str.
  """

  def _act(x, reuse=None, name="ln_out"):
    """Layer-normalized activation function.

    Args:
        x: Input tensor.
        reuse: Whether to reuse the parameters, bool.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    with tf.variable_scope(scope + "_params", reuse=reuse):
      if affine:
        x_shape = [x.get_shape()[-1]]
        beta = nn.weight_variable(
            x_shape,
            init_method="constant",
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            x_shape,
            init_method="constant",
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
    x_normed, x_mean = layer_norm(
        x,
        axes=[1],
        gamma=gamma,
        beta=beta,
        eps=eps,
        scope=scope,
        name=name,
        return_mean=True)
    if l1_reg > 0.0:
      tf.add_to_collection(L1_REG_KEY,
                           l1_reg * tf.reduce_mean(tf.abs(x - x_mean)))
    return act(x_normed)

  return _act


def get_dn_act(act,
               sum_window=30,
               sup_window=30,
               affine=False,
               eps=1.0,
               l1_reg=0.0,
               scope="dn_act"):
  """Gets a divisive-normalized activation function.

  Args:
      act: Activation function, callable object.
      sum_window: Summation window size, [H_sum, W_sum].
      sup_winow: Suppression window size, [H_sup, W_sup].
      affine: Whether to add affine transformation, bool.
      eps: Denominator bias, float.
      l1_reg: L1 regularization constant, on the centered pre-activation.
      scope: Scope of the operation, str.
  """

  def _act(x, reuse=None, name="dn_out"):
    """Divisive-normalized activation function.

    Args:
        x: Input tensor.
        reuse: Whether to reuse the parameters, bool.
        name: Name for the output tensor.

    Returns:
        normed: Output tensor.
    """
    with tf.variable_scope(scope + "_params", reuse=reuse):
      if affine:
        x_shape = [x.get_shape()[-1]]
        beta = nn.weight_variable(
            x_shape,
            init_method="constant",
            init_param={"val": 0.0},
            name="beta")
        gamma = nn.weight_variable(
            x_shape,
            init_method="constant",
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
    x_normed, x_mean = div_norm_1d(
        x,
        sum_window,
        sup_window,
        gamma=gamma,
        beta=beta,
        eps=eps,
        scope=scope,
        return_mean=True,
        name=name)
    if l1_reg > 0.0:
      tf.add_to_collection(L1_REG_KEY,
                           l1_reg * tf.reduce_mean(tf.abs(x - x_mean)))
    return act(x_normed)

  return _act


def get_tf_fn(name):
  """Gets a tensorflow function."""
  if name == "relu":
    return tf.nn.relu
  elif name == "sigmoid":
    return tf.sigmoid
  elif name == "tanh":
    return tf.tanh
  else:
    raise Exception("Unknown tf function \"{}\"".format(name))


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


def _linear(args, output_size, bias, init_scale=0.1, bias_start=0.0,
            scope=None):
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", [total_arg_size, output_size],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(
            -init_scale, init_scale, dtype=dtype))
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(nn.concat(args, 1), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term


if tf.__version__.startswith("0"):
  BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
  BasicRNNCell = tf.nn.rnn_cell.BasicRNNCell
  LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
  MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
  seq2seq_loss = tf.nn.seq2seq.sequence_loss_by_example
else:
  BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell
  BasicRNNCell = tf.contrib.rnn.BasicRNNCell
  LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple
  MultiRNNCell = tf.contrib.rnn.MultiRNNCell
  seq2seq_loss = tf.contrib.seq2seq.sequence_loss


class RNNSimpleCell(BasicRNNCell):
  """The most basic RNN cell."""

  def __init__(self,
               num_units,
               input_size=None,
               activation=tf.tanh,
               init_scale=0.1):
    super(RNNSimpleCell, self).__init__(
        num_units, input_size=input_size, activation=activation)
    self.init_scale = init_scale

  def __call__(self,
               inputs,
               state,
               scope=None,
               reuse=None,
               reuse_bn=None,
               is_training=True):
    with tf.variable_scope(scope or type(self).__name__):  # "RNNSimpleCell"
      with tf.variable_scope("RNN_weights", reuse=reuse):
        z = _linear(
            [inputs, state],
            self._num_units,
            True,
            scope="Linear",
            init_scale=self.init_scale)
      output = self._activation(z, reuse=reuse)
    return output, output


class LSTMSimpleCell(BasicLSTMCell):

  def __init__(self,
               num_units,
               forget_bias=0.0,
               input_size=None,
               init_scale=0.1,
               state_is_tuple=True,
               gate_activation=tf.sigmoid,
               state_activation=tf.tanh):
    super(LSTMSimpleCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        input_size=input_size,
        state_is_tuple=state_is_tuple)
    self.init_scale = init_scale
    self.gate_activation = gate_activation
    self.state_activation = state_activation

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    with tf.variable_scope(scope or type(self).__name__):  # "LSTMSimpleCell"
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = nn.split(state, 2, 1)
      with tf.variable_scope("LSTM_weights", reuse=reuse):
        concat = _linear(
            [inputs, h],
            4 * self._num_units,
            True,
            scope="Linear",
            init_scale=self.init_scale)
      i, j, f, o = nn.split(concat, 4, 1)
      with tf.variable_scope("gate_i"):
        i = self.gate_activation(i, reuse=reuse)
      with tf.variable_scope("gate_f"):
        f = self.gate_activation(f + self._forget_bias, reuse=reuse)
      with tf.variable_scope("state_z"):
        z = self.state_activation(j, reuse=reuse)
      new_c = c * f + i * z
      with tf.variable_scope("state_c"):
        c2 = self.state_activation(new_c, reuse=reuse)
      with tf.variable_scope("gate_o"):
        o = self.gate_activation(o, reuse=reuse)
      new_h = o * c2
      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = nn.concat([new_c, new_h], 1)
      return new_h, new_state


class RNNBNCell(BasicRNNCell):
  """The most basic RNN cell."""

  def __init__(self,
               num_units,
               input_size=None,
               activation=tf.tanh,
               init_scale=0.1,
               eps=1e-3,
               affine=True,
               l1_reg=0.0):
    super(RNNBNCell, self).__init__(
        num_units, input_size=input_size, activation=activation)
    self.affine = affine
    self.eps = eps
    self.init_scale = init_scale
    self.l1_reg = l1_reg
    self.unroll_count = -1

  def __call__(self,
               inputs,
               state,
               scope=None,
               reuse=None,
               reuse_bn=None,
               is_training=True):
    self.unroll_count += 1
    g = tf.get_default_graph()
    with tf.variable_scope(scope or type(self).__name__):  # "RNNSimpleCell"
      with tf.variable_scope("RNN_weights", reuse=reuse):
        i2h = _linear(
            [inputs],
            self._num_units,
            True,
            scope="LinearI",
            init_scale=self.init_scale)
        h2h = _linear(
            [state],
            self._num_units,
            True,
            scope="LinearH",
            init_scale=self.init_scale)
        with tf.variable_scope("bn_params"):
          if self.affine:
            beta_i = nn.weight_variable(
                [self._num_units],
                init_method="constant",
                init_param={"val": 0.0},
                name="beta_i")
            gamma_i = nn.weight_variable(
                [self._num_units],
                init_method="constant",
                init_param={"val": 1.0},
                name="gamma_i")
            beta_h = nn.weight_variable(
                [self._num_units],
                init_method="constant",
                init_param={"val": 0.0},
                name="beta_h")
            gamma_h = nn.weight_variable(
                [self._num_units],
                init_method="constant",
                init_param={"val": 1.0},
                name="gamma_h")
          else:
            beta_i = None
            gamma_i = None
            beta_h = None
            gamma_h = None
      i2h_norm, mean_i = batch_norm(
          i2h,
          self._num_units,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_i,
          beta=beta_i,
          axes=[0],
          eps=self.eps,
          scope="bn_i_{}".format(self.unroll_count),
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY,
                             self.l1_reg * tf.reduce_mean(tf.abs(i2h - mean_i)))
      h2h_norm, mean_h = batch_norm(
          h2h,
          self._num_units,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_h,
          beta=beta_h,
          axes=[0],
          eps=self.eps,
          scope="bn_h_{}".format(self.unroll_count),
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY,
                             self.l1_reg * tf.reduce_mean(tf.abs(h2h - mean_h)))
      output = self._activation(i2h_norm + h2h_norm)
    return output, output


class LSTMLNCell(BasicLSTMCell):

  def __init__(self,
               num_units,
               forget_bias=0.0,
               input_size=None,
               init_scale=0.1,
               state_is_tuple=True,
               activation=tf.tanh,
               dropout_after=False,
               eps=1e-3,
               affine=True,
               gate_activation=tf.sigmoid,
               state_activation=tf.tanh,
               l1_reg=0.0):
    super(LSTMLNCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        input_size=input_size,
        state_is_tuple=state_is_tuple)
    self.eps = eps
    self.affine = affine
    self.init_scale = init_scale
    self.gate_activation = gate_activation
    self.state_activation = state_activation
    self.l1_reg = l1_reg

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    with tf.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = nn.split(state, 2, 1)
      with tf.variable_scope("LSTM_weights", reuse=reuse):
        i2h = _linear(
            [inputs],
            4 * self._num_units,
            True,
            scope="LinearI",
            init_scale=self.init_scale)
        h2h = _linear(
            [h],
            4 * self._num_units,
            True,
            scope="LinearH",
            init_scale=self.init_scale)
        if self.affine:
          with tf.variable_scope("ln_params"):
            beta_i = nn.weight_variable(
                [4 * self._num_units],
                init_method="constant",
                init_param={"val": 0.0},
                name="beta_i")
            gamma_i = nn.weight_variable(
                [4 * self._num_units],
                init_method="constant",
                init_param={"val": 1.0},
                name="gamma_i")
            beta_h = nn.weight_variable(
                [4 * self._num_units],
                init_method="constant",
                init_param={"val": 0.0},
                name="beta_h")
            gamma_h = nn.weight_variable(
                [4 * self._num_units],
                init_method="constant",
                init_param={"val": 1.0},
                name="gamma_h")
            beta_c = nn.weight_variable(
                [self._num_units],
                init_method="constant",
                init_param={"val": 0.0},
                name="beta_c")
            gamma_c = nn.weight_variable(
                [self._num_units],
                init_method="constant",
                init_param={"val": 1.0},
                name="gamma_c")
        else:
          beta_i = None
          gamma_i = None
          beta_h = None
          gamma_h = None
          beta_c = None
          gamma_c = None
      i2h_norm, mean_i = layer_norm(
          i2h,
          gamma=gamma_i,
          beta=beta_i,
          eps=self.eps,
          axes=[1],
          scope="ln_i",
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY,
                             self.l1_reg * tf.reduce_mean(tf.abs(i2h - mean_i)))
      h2h_norm, mean_h = layer_norm(
          h2h,
          gamma=gamma_h,
          beta=beta_h,
          axes=[1],
          eps=self.eps,
          scope="ln_h",
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY,
                             self.l1_reg * tf.reduce_mean(tf.abs(h2h - mean_h)))
      concat = i2h_norm + h2h_norm

      i, j, f, o = nn.split(concat, 4, 1)
      new_c = (c * self.gate_activation(f + self._forget_bias) +
               self.gate_activation(i) * self.state_activation(j))

      # Second normalization before tanh.
      new_c_norm, mean_c = layer_norm(
          new_c,
          gamma=gamma_c,
          beta=beta_c,
          eps=self.eps,
          axes=[1],
          scope="ln_c",
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY, self.l1_reg *
                             tf.reduce_mean(tf.abs(new_c - mean_c)))
      new_h = self.state_activation(new_c_norm) * self.gate_activation(o)
      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c_norm, new_h)
      else:
        new_state = nn.concat([new_c_norm, new_h], 1)
    return new_h, new_state


class LSTMBNCell(BasicLSTMCell):

  def __init__(self,
               num_units,
               forget_bias=0.0,
               input_size=None,
               init_scale=0.1,
               state_is_tuple=True,
               eps=1e-3,
               affine=True,
               keep_prob=1.0,
               gate_activation=tf.sigmoid,
               state_activation=tf.tanh,
               l1_reg=0.0):
    super(LSTMBNCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        input_size=input_size,
        state_is_tuple=state_is_tuple)
    self.eps = eps
    self.affine = affine
    self.init_scale = init_scale
    self.unroll_count = -1
    self.keep_prob = keep_prob
    self.gate_activation = gate_activation
    self.state_activation = state_activation
    self.l1_reg = l1_reg

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    self.unroll_count += 1
    with tf.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = nn.split(state, 2, 1)
      with tf.variable_scope("LSTM_weights", reuse=reuse):
        i2h = _linear(
            [inputs],
            4 * self._num_units,
            True,
            scope="LinearI",
            init_scale=self.init_scale)
        h2h = _linear(
            [h],
            4 * self._num_units,
            True,
            scope="LinearH",
            init_scale=self.init_scale)
        beta_i = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_i")
        gamma_i = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 1.0},
            name="gamma_i")
        beta_h = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_h")
        gamma_h = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 1.0},
            name="gamma_h")
        beta_c = nn.weight_variable(
            [self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_c")
        gamma_c = nn.weight_variable(
            [self._num_units],
            init_method="constant",
            init_param={"val": 1.0},
            name="gamma_c")
      i2h_norm, mean_i = batch_norm(
          i2h,
          self._num_units * 4,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_i,
          beta=beta_i,
          axes=[0],
          eps=self.eps,
          scope="bn_i_{}".format(self.unroll_count),
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY,
                             self.l1_reg * tf.reduce_mean(tf.abs(i2h - mean_i)))
      h2h_norm, mean_h = batch_norm(
          h2h,
          self._num_units * 4,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_h,
          beta=beta_h,
          axes=[0],
          eps=self.eps,
          scope="bn_h_{}".format(self.unroll_count),
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY,
                             self.l1_reg * tf.reduce_mean(tf.abs(h2h - mean_h)))
      i, j, f, o = nn.split(i2h_norm + h2h_norm, 4, 1)
      new_c = (c * self.gate_activation(f + self._forget_bias) +
               self.gate_activation(i) * self.state_activation(j))
      new_c_norm, mean_c = batch_norm(
          new_c,
          self._num_units,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_c,
          beta=beta_c,
          axes=[0],
          eps=self.eps,
          scope="bn_c_{}".format(self.unroll_count),
          return_mean=True)
      if self.l1_reg > 0.0:
        tf.add_to_collection(L1_REG_KEY, self.l1_reg *
                             tf.reduce_mean(tf.abs(new_c - mean_c)))
      new_h = self.state_activation(new_c_norm) * self.gate_activation(o)
      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c_norm, new_h)
      else:
        new_state = nn.concat([new_c_norm, new_h], 1)
    return new_h, new_state


class MultiRNNNormCell(MultiRNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("Cell%d" % i):
          if self._state_is_tuple:
            cur_state = state[i]
          else:
            cur_state = tf.slice(state, [0, cur_state_pos],
                                 [-1, cell.state_size])
            cur_state_pos += cell.state_size
          cur_inp, new_state = cell(
              cur_inp,
              cur_state,
              is_training=is_training,
              reuse=reuse,
              reuse_bn=reuse_bn)
          new_states.append(new_state)
    new_states = (tuple(new_states)
                  if self._state_is_tuple else nn.concat(new_states, 1))
    return cur_inp, new_states


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_
    self._config = config

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    cells = []

    for ii in range(config.num_layers):
      if config.rnn_cell == "lstm":
        if config.norm_field == "batch":
          log.info("Applying batch normalization on LSTM")
          log.info("Setting eps={:.3e}".format(config.sigma_init**2))
          log.info("Setting L1={:.3e}".format(config.l1_reg))
          log.info("Setting affine={}".format(config.norm_affine))
          rnn_cell = LSTMBNCell(
              size,
              forget_bias=0.0,
              state_is_tuple=True,
              init_scale=config.init_scale,
              eps=config.sigma_init**2,
              affine=config.norm_affine,
              gate_activation=get_tf_fn(config.gate_act_fn),
              state_activation=get_tf_fn(config.state_act_fn),
              l1_reg=config.l1_reg)
        elif config.norm_field == "layer" and not config.norm_activation:
          log.info("Applying layer normalization on LSTM")
          log.info("Setting eps={:.3e}".format(config.sigma_init**2))
          log.info("Setting L1={:.3e}".format(config.l1_reg))
          log.info("Setting affine={}".format(config.norm_affine))
          rnn_cell = LSTMLNCell(
              size,
              forget_bias=0.0,
              state_is_tuple=True,
              init_scale=config.init_scale,
              eps=config.sigma_init**2,
              affine=config.norm_affine,
              gate_activation=get_tf_fn(config.gate_act_fn),
              state_activation=get_tf_fn(config.state_act_fn),
              l1_reg=config.l1_reg)
        else:
          if config.norm_field == "layer":
            log.info("Applying layer normalization on LSTM activations")
            log.info("Setting eps={:.3e}".format(config.sigma_init**2))
            log.info("Setting L1={:.3e}".format(config.l1_reg))
            log.info("Setting affine={}".format(config.norm_affine))
            gate_act = get_ln_act(
                get_tf_fn(config.gate_act_fn),
                eps=config.sigma_init**2,
                l1_reg=config.l1_reg,
                affine=config.norm_affine)
            state_act = get_ln_act(
                get_tf_fn(config.state_act_fn),
                eps=config.sigma_init**2,
                l1_reg=config.l1_reg,
                affine=config.norm_affine)
          elif config.norm_field == "div":
            log.info("Applying divisive normalization on LSTM activations")
            log.info("Setting eps={:.3e}".format(config.sigma_init**2))
            log.info("Setting L1={:.3e}".format(config.l1_reg))
            log.info("Setting affine={}".format(config.norm_affine))
            gate_act = get_dn_act(
                get_tf_fn(config.gate_act_fn),
                sum_window=config.dn_window,
                sup_window=config.dn_window,
                eps=config.sigma_init**2,
                l1_reg=config.l1_reg,
                affine=config.norm_affine)
            state_act = get_dn_act(
                get_tf_fn(config.state_act_fn),
                sum_window=config.dn_window,
                sup_window=config.dn_window,
                eps=config.sigma_init**2,
                l1_reg=config.l1_reg,
                affine=config.norm_affine)
          elif config.norm_field == "no":
            log.info("Baseline LSTM")
            log.info("Setting L1={:.3e}".format(config.l1_reg))
            gate_act = get_reg_act(
                get_tf_fn(config.gate_act_fn), l1_reg=config.l1_reg)
            state_act = get_reg_act(
                get_tf_fn(config.state_act_fn), l1_reg=config.l1_reg)
          else:
            raise Exception("Unknown normalization field.")
          rnn_cell = LSTMSimpleCell(
              size,
              forget_bias=0.0,
              state_is_tuple=True,
              init_scale=config.init_scale,
              gate_activation=gate_act,
              state_activation=state_act)
      else:
        if config.norm_field == "batch":
          log.info("Applying batch normalization on RNN")
          log.info("Setting eps={:.3e}".format(config.sigma_init**2))
          log.info("Setting L1={:.3e}".format(config.l1_reg))
          log.info("Setting affine={}".format(config.norm_affine))
          rnn_cell = RNNBNCell(
              size,
              init_scale=config.init_scale,
              activation=get_tf_fn(config.state_act_fn),
              affine=config.norm_affine,
              eps=config.sigma_init**2,
              l1_reg=config.l1_reg)
        else:
          if config.norm_field == "layer":
            log.info("Applying layer normalization on RNN")
            log.info("Setting eps={:.3e}".format(config.sigma_init**2))
            log.info("Setting L1={:.3e}".format(config.l1_reg))
            log.info("Setting affine={}".format(config.norm_affine))
            state_act = get_ln_act(
                get_tf_fn(config.state_act_fn),
                eps=config.sigma_init**2,
                l1_reg=config.l1_reg,
                affine=config.norm_affine)
          elif config.norm_field == "div":
            log.info("Applying divisive normalization on RNN")
            log.info("Setting eps={:.3e}".format(config.sigma_init**2))
            log.info("Setting L1={:.3e}".format(config.l1_reg))
            log.info("Setting affine={}".format(config.norm_affine))
            state_act = get_dn_act(
                get_tf_fn(config.state_act_fn),
                sum_window=config.dn_window,
                sup_window=config.dn_window,
                eps=config.sigma_init**2,
                l1_reg=config.l1_reg,
                affine=config.norm_affine)
          elif config.norm_field == "no":
            log.info("Baseline RNN")
            log.info("Setting L1={:.3e}".format(config.l1_reg))
            state_act = get_reg_act(
                get_tf_fn(config.state_act_fn), l1_reg=config.l1_reg)
          rnn_cell = RNNSimpleCell(
              size, init_scale=config.init_scale, activation=state_act)

      cells.append(rnn_cell)
    cell = MultiRNNNormCell(cells, state_is_tuple=True)
    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size],
          dtype=data_type(),
          initializer=tf.random_uniform_initializer(
              -config.init_scale, config.init_scale, dtype=data_type()))
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        reuse = time_step > 0 or not is_training
        if is_training:
          reuse_bn = None
        else:
          reuse_bn = True
        (cell_output, state) = cell(
            inputs[:, time_step, :],
            state,
            is_training=is_training,
            reuse=reuse,
            reuse_bn=reuse_bn)
        outputs.append(cell_output)

    output = tf.reshape(nn.concat(outputs, 1), [-1, size])

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size],
        dtype=data_type(),
        initializer=tf.random_uniform_initializer(
            -config.init_scale, config.init_scale, dtype=data_type()))
    softmax_b = tf.get_variable(
        "softmax_b", [vocab_size],
        dtype=data_type(),
        initializer=tf.constant_initializer(
            0.0, dtype=data_type()))
    logits = tf.matmul(output, softmax_w) + softmax_b

    if tf.__version__.startswith("0"):
      loss = seq2seq_loss(
          [logits], [tf.reshape(input_.targets, [-1])],
          [tf.ones(
              [batch_size * num_steps], dtype=data_type())])
    else:
      loss = seq2seq_loss(
          tf.expand_dims(logits, 0),
          tf.reshape(input_.targets, [1, -1]),
          tf.ones(
              [1, batch_size * num_steps], dtype=data_type()),
          average_across_timesteps=False,
          average_across_batch=False)
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    if not is_training:
      return

    regularizer = tf.get_collection(L1_REG_KEY)
    if len(regularizer) > 0:
      log.info("Regularizer variables {}".format(regularizer))
      _opt_cost = tf.add_n(regularizer) + cost
    else:
      _opt_cost = cost

    self._lr = tf.Variable(0.0, trainable=False, name="learn_rate")
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(_opt_cost, tvars), config.max_grad_norm)
    if config.optimizer == "gd":
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
    elif config.optimizer == "adam":
      optimizer = tf.train.AdamOptimizer(self._lr)
    else:
      raise Exception("Unknown optimizer")

    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def config(self):
    return self._config

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def run_epoch(session, model, epoch, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = []
  if model.config.rnn_cell == "lstm":
    for ss, st in enumerate(model.initial_state):
      state.append({})
      state[ss]["c"] = session.run(st.c)
      state[ss]["h"] = session.run(st.h)
  elif model.config.rnn_cell == "rnn":
    for ss, st in enumerate(model.initial_state):
      state.append(session.run(st))

  fetch_list = [model.cost]
  if model.config.rnn_cell == "lstm":
    for ss, st in enumerate(model.final_state):
      fetch_list.append(st.c)
      fetch_list.append(st.h)
  else:
    for ss, st in enumerate(model.final_state):
      fetch_list.append(st)

  if eval_op is not None:
    fetch_list.append(eval_op)

  for step in range(model.input.epoch_size):
    feed_dict = {}
    if model.config.rnn_cell == "lstm":
      for ss, st in enumerate(model.initial_state):
        feed_dict[st.c] = state[ss]["c"]
        feed_dict[st.h] = state[ss]["h"]
    else:
      for ss, st in enumerate(model.initial_state):
        feed_dict[st] = state[ss]

    results = session.run(fetch_list, feed_dict)
    cost = results[0]
    state_results = results[1:]
    state = []
    if model.config.rnn_cell == "lstm":
      for ss, st in enumerate(model.final_state):
        state.append({})
        state[ss] = {"c": state_results[2 * ss], "h": state_results[2 * ss + 1]}
    else:
      for ss, st in enumerate(model.final_state):
        state.append(state_results[ss])

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      ppl = np.exp(costs / iters)
      speed = iters * model.input.batch_size / (time.time() - start_time)
      if model.config.level == "word":
        cost_str = "Perplexity = {:8.3f}".format(ppl)
        speed_str = "Speed = {:6.0f} WPS".format(speed)
      elif model.config.level == "char":
        bpc = np.log2(ppl)
        cost_str = "BPC = {:8.4f}".format(bpc)
        speed_str = "Speed = {:6.0f} CPS".format(speed)
      log.info("Epoch = {:5.1f} || {} || {}".format(
          epoch + step / model.input.epoch_size, cost_str, speed_str))

  if model.config.level == "word":
    return np.exp(costs / iters)
  elif model.config.level == "char":
    return np.log2(np.exp(costs / iters))


def get_config():
  if FLAGS.config is not None:
    return conf.BaselineLSTMConfig.from_json(open(FLAGS.config, "r").read())
  else:
    return conf.get_config(FLAGS.model)


def get_environ():
  """Gets an environment object."""
  # Manually set environment.
  if FLAGS.env is not None:
    return BasicEnvironment.from_json(open(FLAGS.env, "r").read())

  if FLAGS.data_folder is None:
    data_folder = "../data/ptb"
  else:
    data_folder = FLAGS.data_folder
  exp_id = "exp_ptb" + "_" + FLAGS.model
  if FLAGS.id is None:
    exp_id = gen_id(exp_id)
  else:
    exp_id = FLAGS.id
  return BasicEnvironment(
      device=get_device(FLAGS.gpu),
      dataset="ptb",
      data_folder=data_folder,
      logs_folder=FLAGS.logs,
      save_folder=FLAGS.results,
      verbose=FLAGS.verbose,
      exp_id=exp_id,
      description=FLAGS.description)


class ExperimentLogger():

  def __init__(self, logs_folder, level="word"):
    """Initialize files."""
    if not os.path.isdir(logs_folder):
      os.makedirs(logs_folder)

    catalog_file = os.path.join(logs_folder, "catalog")

    with open(catalog_file, "w") as f:
      f.write("filename,type,name\n")

    with open(catalog_file, "a") as f:
      f.write("{},plain,{}\n".format("cmd.txt", "Commands"))

    with open(os.path.join(logs_folder, "cmd.txt"), "w") as f:
      f.write(" ".join(sys.argv))

    if level == "word":
      self.cost_str = "Perplexity"
    elif level == "char":
      self.cost_str = "BPC"
    else:
      raise Exception("Unknown level")

    with open(catalog_file, "a") as f:
      f.write("train_cost.csv,csv," + "Train " + self.cost_str + "\n")
      f.write("valid_cost.csv,csv," + "Valid " + self.cost_str + "\n")
      f.write("test_cost.csv,csv," + "Test " + self.cost_str + "\n")
      f.write("learn_rate.csv,csv,Learning Rate\n")

    self.train_file_name = os.path.join(logs_folder, "train_cost.csv")
    if not os.path.exists(self.train_file_name):
      with open(self.train_file_name, "w") as f:
        f.write("step,time,cost\n")

    self.valid_file_name = os.path.join(logs_folder, "valid_cost.csv")
    if not os.path.exists(self.valid_file_name):
      with open(self.valid_file_name, "w") as f:
        f.write("step,time,cost\n")

    self.test_file_name = os.path.join(logs_folder, "test_cost.csv")
    if not os.path.exists(self.test_file_name):
      with open(self.test_file_name, "w") as f:
        f.write("step,time,cost\n")

    self.learn_rate_file_name = os.path.join(logs_folder, "learn_rate.csv")
    if not os.path.exists(self.learn_rate_file_name):
      with open(self.learn_rate_file_name, "w") as f:
        f.write("step,time,lr\n")

  def log_train_cost(self, epoch, cost):
    """Writes training cost."""
    log.info("Epoch = {:5.1f} || Train {} = {:8.3f}".format(
        epoch + 1, self.cost_str, cost))
    with open(self.train_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          epoch + 1, datetime.datetime.now().isoformat(), cost))

  def log_valid_cost(self, epoch, cost):
    """Writes validation cost."""
    log.info("Epoch = {:5.1f} || Valid {} = {:8.3f}".format(
        epoch + 1, self.cost_str, cost))
    with open(self.valid_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          epoch + 1, datetime.datetime.now().isoformat(), cost))

  def log_test_cost(self, epoch, cost):
    """Writes test cost."""
    log.info("Epoch = {:5.1f} || Test {} = {:8.3f}".format(epoch + 1,
                                                           self.cost_str, cost))
    with open(self.test_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          epoch + 1, datetime.datetime.now().isoformat(), cost))

  def log_learn_rate(self, epoch, lr):
    """Writes learning rate."""
    log.info("Epoch = {:5.1f} || Learning Rate = {:.3e}".format(epoch + 1, lr))
    with open(self.learn_rate_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          epoch + 1, datetime.datetime.now().isoformat(), lr))


def main(_):
  if not FLAGS.data_folder:
    raise ValueError("Must set --data_folder to PTB data directory")

  config = get_config()
  environ = get_environ()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  log.info("Environment: {}".format(environ.__dict__))
  log.info("Config: {}".format(config.__dict__))

  if environ.verbose:
    verbose_level = 0
  else:
    verbose_level = 2

  with log.verbose_level(verbose_level):
    raw_data = reader.ptb_raw_data(FLAGS.data_folder, level=config.level)
    train_data, valid_data, test_data, _ = raw_data
    save_folder = os.path.join(environ.save_folder, environ.exp_id)
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)

    config_file = os.path.join(save_folder, "conf.json")
    environ_file = os.path.join(save_folder, "env.json")
    with open(config_file, "w") as f:
      f.write(config.to_json())
    with open(environ_file, "w") as f:
      f.write(environ.to_json())

    with tf.Graph().as_default():
      tf.set_random_seed(1234)
      with tf.name_scope("Train"):
        train_input = PTBInput(
            config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None):
          m = PTBModel(is_training=True, config=config, input_=train_input)
      with tf.name_scope("Valid"):
        valid_input = PTBInput(
            config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True):
          mvalid = PTBModel(
              is_training=False, config=config, input_=valid_input)
      with tf.name_scope("Test"):
        test_input = PTBInput(config=config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True):
          mtest = PTBModel(
              is_training=False, config=eval_config, input_=test_input)

      sv = tf.train.Supervisor(logdir=save_folder)
      exp_logger = ExperimentLogger(
          os.path.join(environ.logs_folder, environ.exp_id), level=config.level)
      with sv.managed_session() as session:
        if environ.verbose:
          loop = range(config.max_max_epoch)
        else:
          loop = pb.get(config.max_max_epoch)
        for epoch in loop:
          lr_decay = config.lr_decay**max(epoch - config.max_epoch + 1, 0.0)
          lr = config.learning_rate * lr_decay
          m.assign_lr(session, lr)
          log.info("Experiment ID {}".format(environ.exp_id))
          train_cost = run_epoch(
              session, m, epoch, eval_op=m.train_op, verbose=True)
          valid_cost = run_epoch(session, mvalid, epoch)
          exp_logger.log_train_cost(epoch, train_cost)
          exp_logger.log_valid_cost(epoch, valid_cost)
          exp_logger.log_learn_rate(epoch, lr)
        if config.max_max_epoch > 0:
          test_cost = run_epoch(session, mtest, epoch)
          exp_logger.log_test_cost(epoch, test_cost)
          log.info("Saving model to {}.".format(save_folder))
          sv.saver.save(session, save_folder, global_step=sv.global_step)
          log.info("Final test cost = {:8.3f}".format(test_cost))


if __name__ == "__main__":
  tf.app.run()
