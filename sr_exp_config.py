from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import os
from cifar_exp_config import BaselineConfig


def get_config(dataset, model):
  if model == "base":
    config = BaselineSRConfig()
  elif model == "dn":
    config = DNConfig()
  elif model == "dnms":
    config = DNMSConfig()
  elif model == "dn-star":
    config = DNStarSRConfig()
  else:
    raise Exception("Unknown model \"{}\"".format(model))

  return config


class BaselineSRConfig(BaselineConfig):

  def __init__(self):
    self.model = "base"
    self.batch_size = 1
    self.height = 33
    self.width = 33
    self.num_channel = 1
    self.disp_iter = 1
    self.save_iter = 100000
    self.valid_iter = 2500
    self.max_train_iter = 100000000
    self.momentum = 0.9
    self.base_learn_rate = 1e-4
    self.filter_size = [[9, 9, 1, 64], [5, 5, 64, 32], [5, 5, 32, 1]]
    self.strides = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    self.pool_fn = [None, None, None]
    self.pool_size = [[], [], []]
    self.pool_strides = [[], [], []]
    self.conv_act_fn = ["relu", "relu", None]
    self.conv_init_method = None
    self.conv_init_std = [1.0e-3, 1.0e-3, 1.0e-3]
    self.lr_decay_steps = []
    self.wd = 0.0
    self.norm_field = None
    self.stagewise_norm = False
    self.sigma_init = 1e-2
    self.learn_sigma = False
    self.norm_affine = False
    self.l1_reg = 0.0
    self.prefetch = False
    self.data_aug = False
    self.whiten = False
    self.div255 = False
    self.resize_factor = 3.0
    self.patch_size = 33
    self.num_patch_per_img = 100
    self.dataset = "HDF5Matlab"
    self.crop_border = 8


class DNConfig(BaselineSRConfig):

  def __init__(self):
    super(DNMSConfig, self).__init__()

    self.l1_reg = 1.0e-1
    self.sigma_init = 1.0
    self.norm_field = "div"
    self.sum_window = [[5, 5], [5, 5], None]
    self.sup_window = None
    self.norm_affine = False
    self.model = "dn"


class DNMSConfig(BaselineSRConfig):

  def __init__(self):
    super(DNMSConfig, self).__init__()

    self.l1_reg = 1.0e-1
    self.sigma_init = 1.0
    self.norm_field = "div_ms"
    self.sum_window = [[5, 5], [5, 5], None]
    self.sup_window = None
    self.norm_affine = False
    self.model = "dnms"


class DNStarSRConfig(BaselineSRConfig):

  def __init__(self):
    super(DNStarSRConfig, self).__init__()

    self.l1_reg = 1.0e-1
    self.sigma_init = 1.0
    self.norm_field = "div"
    self.norm_affine = False
    self.sum_window = [[5, 5], [5, 5], None]
    self.sup_window = [[7, 7], [7, 7], None]
    self.model = "dn-star"