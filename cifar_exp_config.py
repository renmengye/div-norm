from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import os


def get_config(dataset, model):
  # Use one of the pre-set config.
  if model == "base":
    config = BaselineConfig()
  elif model == "base-drop":
    config = BaselineDropoutConfig()
  elif model == "base-l1":
    config = BaselineConfig()
    config.model = "base-l1"
    if dataset == "cifar-10":
      config.l1_reg = 0.01
    else:
      config.l1_reg = 0.01
  elif model == "bn":
    config = BNConfig()
  elif model == "bnms":
    config = BNMSConfig()
  elif model == "bn-s":
    config = BNSConfig()
    if dataset == "cifar-10":
      config.sigma_init = 1.0
    else:
      config.sigma_init = 0.25
  elif model == "bn-l1":
    config = BNL1Config()
    if dataset == "cifar-10":
      config.l1_reg = 0.005
    else:
      config.l1_reg = 0.01
  elif model == "bn-star":
    config = BNStarConfig()
    if dataset == "cifar-10":
      config.sigma_init = 1.0
      config.l1_reg = 0.005
    else:
      config.sigma_init = 0.25
      config.l1_reg = 0.01
  elif model == "ln":
    config = LNConfig()
  elif model == "lnms":
    config = LNMSConfig()
  elif model == "ln-s":
    config = LNSConfig()
    if dataset == "cifar-10":
      config.sigma_init = 0.5
    else:
      config.sigma_init = 0.5
  elif model == "ln-l1":
    config = LNL1Config()
    if dataset == "cifar-10":
      config.l1_reg = 0.005
    else:
      config.l1_reg = 0.005
  elif model == "ln-star":
    config = LNStarConfig()
    if dataset == "cifar-10":
      config.sigma_init = 0.5
      config.l1_reg = 0.005
    else:
      config.sigma_init = 0.5
      config.l1_reg = 0.005
  elif model == "dn":
    config = DNConfig()
    if dataset == "cifar-10":
      config.sigma_init = 0.5
    else:
      config.sigma_init = 0.5
  elif model == "dnms":
    config = DNMSConfig()
  elif model == "dn-star":
    config = DNStarConfig()
    if dataset == "cifar-10":
      config.sigma_init = 0.5
      config.l1_reg = 0.005
    else:
      config.sigma_init = 0.5
      config.l1_reg = 0.001
  elif model == "resnet-32":
    config = ResNet32Config()
  elif model == "resnet-32-no":
    config = ResNet32NoNormConfig()
  elif model == "resnet-32-bn":
    config = ResNet32BNConfig()
  elif model == "resnet-32-dn":
    config = ResNet32DNConfig()
    if dataset == "cifar-10":
      config.sigma_init = 0.5
    else:
      config.sigma_init = 2.0
  elif model == "resnet-32-dn-star":
    config = ResNet32DNConfig()
    if dataset == "cifar-10":
      config.sigma_init = 0.5
      config.l1_reg = 1e-3
    else:
      config.sigma_init = 2.0
      config.l1_reg = 1e-2
  elif model == "resnet-32-ln":
    config = ResNet32LNConfig()
  elif model == "resnet-110":
    config = ResNet110Config()
  elif model == "resnet-164":
    config = ResNet164Config()
  else:
    raise Exception("Unknown model \"{}\"".format(model))
  if dataset == "cifar-10":
    config.mlp_dims = [1024, 64, 10]
    config.num_classes = 10
  elif dataset == "cifar-100":
    config.mlp_dims = [1024, 64, 100]
    config.num_classes = 100
  else:
    raise Exception("Unknown dataset")
  return config


class BaselineConfig(object):
  """Standard CNN on CIFAR-10"""

  def __init__(self):
    self.model = "base"
    self.batch_size = 100
    self.height = 32
    self.width = 32
    self.num_channel = 3
    self.disp_iter = 100
    self.save_iter = 5000
    self.valid_iter = 500
    self.max_train_iter = 50000
    self.momentum = 0.9
    self.base_learn_rate = 1e-3
    self.label_size = 10
    self.filter_size = [[5, 5, 3, 32], [5, 5, 32, 32], [5, 5, 32, 64]]
    self.strides = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    self.pool_fn = ["max_pool", "avg_pool", "avg_pool"]
    self.pool_size = [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]]
    self.pool_strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]]
    self.conv_act_fn = ["relu"] * 3
    self.conv_init_method = None
    self.conv_init_std = [1.0e-4, 1.0e-2, 1.0e-2]
    self.mlp_init_method = None
    self.mlp_init_std = [1.0e-1, 1.0e-1]
    self.mlp_act_fn = [None] * 2
    self.mlp_dims = [1024, 64, 10]
    self.lr_decay_steps = [5000, 30000]
    self.wd = 0.0
    self.mlp_dropout = [False, False]
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

  def set_name(self, val):
    self.model = val
    return self

  def set_whiten(self, val):
    self.whiten = val
    return self

  def set_div255(self, val):
    self.div255 = val
    return self

  def set_mlp_dropout(self, val):
    self.mlp_dropout = [val] * 2
    return self

  def set_max_train_iter(self, val):
    self.max_train_iter = val
    return self

  def set_lr_decay_steps(self, val):
    self.lr_decay_steps = val
    return self

  def set_wd(self, val):
    self.wd = val
    return self

  def set_l1_reg(self, val):
    self.l1_reg = val
    return self

  def set_sigma_init(self, val):
    self.sigma_init = val
    return self

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)

  @classmethod
  def from_json(cls, s):
    dic = json.loads(s)
    config = cls()
    config.__dict__ = dic
    return config


class BaselineDropoutConfig(BaselineConfig):

  def __init__(self):
    super(BaselineDropoutConfig, self).__init__()
    self.mlp_dropout = [True, True]
    self.model = "base-drop"


class BaselineWDConfig(BaselineConfig):

  def __init__(self):
    super(BaselineWDConfig, self).__init__()
    self.wd = 0.1
    self.model = "base-wd"


class BaselineWDDropoutConfig(BaselineWDConfig):

  def __init__(self):
    super(BaselineWDDropoutConfig, self).__init__()
    self.mlp_dropout = [True, True]
    self.model = "base-wd-drop"


class BNConfig(BaselineConfig):

  def __init__(self):
    super(BNConfig, self).__init__()
    self.norm_field = "batch"
    self.sigma_init = 1e-2
    self.norm_affine = True
    self.max_train_iter = 80000
    self.lr_decay_steps = [30000, 50000]
    self.model = "bn"
    self.bn_mask = [True] * 3


class BNMSConfig(BNConfig):

  def __init__(self):
    super(BNMSConfig, self).__init__()
    self.norm_field = "batch_ms"
    self.model = "bnms"


class BNSConfig(BNConfig):

  def __init__(self):
    super(BNSConfig, self).__init__()
    self.sigma_init = 1.0
    self.model = "bn-s"


class BNL1Config(BNConfig):

  def __init__(self):
    super(BNL1Config, self).__init__()
    self.l1_reg = 0.01
    self.model = "bn-l1"


class BNStarConfig(BNConfig):

  def __init__(self):
    super(BNStarConfig, self).__init__()
    self.l1_reg = 0.01
    self.sigma_init = 1.0
    self.model = "bn-star"


class LNConfig(BNConfig):

  def __init__(self):
    super(LNConfig, self).__init__()
    self.norm_field = "layer"
    self.model = "ln"


class LNMSConfig(LNConfig):

  def __init__(self):
    super(LNMSConfig, self).__init__()
    self.norm_field = "layer_ms"
    self.model = "lnms"


class LNSConfig(BNSConfig):

  def __init__(self):
    super(LNSConfig, self).__init__()
    self.norm_field = "layer"
    self.norm_affine = False
    self.model = "ln-s"


class LNL1Config(BNL1Config):

  def __init__(self):
    super(LNL1Config, self).__init__()
    self.norm_field = "layer"
    self.model = "ln-l1"


class LNStarConfig(BNStarConfig):

  def __init__(self):
    super(LNStarConfig, self).__init__()
    self.norm_field = "layer"
    self.norm_affine = False
    self.model = "ln-star"


class DNConfig(BNConfig):

  def __init__(self):
    super(DNConfig, self).__init__()
    self.norm_field = "div"
    self.model = "dn"
    self.sum_window = [[5, 5], [3, 3], [3, 3]]
    self.sup_window = [[5, 5], [3, 3], [3, 3]]
    self.norm_affine = False


class DNMSConfig(DNConfig):

  def __init__(self):
    super(DNMSConfig, self).__init__()
    self.norm_field = "div_ms"
    self.model = "dnms"
    self.sum_window = [[5, 5], [3, 3], [3, 3]]
    self.sup_window = None
    self.norm_affine = False


class DNStarConfig(BNStarConfig):

  def __init__(self):
    super(DNStarConfig, self).__init__()
    self.norm_field = "div"
    self.norm_affine = False
    self.sum_window = [[5, 5], [3, 3], [3, 3]]
    self.sup_window = [[5, 5], [3, 3], [3, 3]]
    self.model = "dn-star"


class ResNet32Config(object):

  def __init__(self):
    self.batch_size = 100
    self.height = 32
    self.width = 32
    self.num_channel = 3
    self.min_lrn_rate = 0.0001
    self.base_learn_rate = 0.1
    self.num_residual_units = [5, 5, 5]  # ResNet-32
    self.seed = 1234
    self.strides = [1, 2, 2]
    self.activate_before_residual = [True, False, False]
    self.init_stride = 1
    self.init_max_pool = False
    self.init_filter = 3
    self.use_bottleneck = False
    self.filters = [16, 16, 32, 64]
    self.wd = 0.0002
    # self.relu_leakiness = 0.1   # Original TF model has leaky relu.
    self.relu_leakiness = 0.0
    self.optimizer = "mom"
    self.max_train_iter = 80000
    self.lr_decay_steps = [40000, 60000]
    self.model = "resnet-32"
    self.disp_iter = 100
    self.save_iter = 5000
    self.valid_iter = 500
    self.norm_field = None
    self.sigma_init = 1e-2
    self.learn_sigma = False
    self.norm_affine = False
    self.stagewise_norm = False
    self.l1_reg = 0.0
    self.prefetch = True
    self.data_aug = True
    self.whiten = False  # Original TF has whiten.
    self.div255 = True

  def set_name(self, model):
    self.model = model
    return self

  def set_l1_reg(self, val):
    self.l1_reg = val
    return self

  def set_wd(self, val):
    self.wd = val
    return self

  def set_sigma_init(self, val):
    self.sigma_init = val
    return self

  def set_max_train_iter(self, val):
    self.max_train_iter = val
    return self

  def set_lr_decay_steps(self, val):
    self.lr_decay_steps = val
    return self

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)

  @classmethod
  def from_json(cls, s):
    dic = json.loads(s)
    config = cls()
    config.__dict__ = dic
    return config


class ResNet32NoNormConfig(ResNet32Config):

  def __init__(self):
    super(ResNet32NoNormConfig, self).__init__()
    self.model = "resnet-32-no"
    self.norm_field = "no"
    self.base_learn_rate = 0.005


class ResNet32DNConfig(ResNet32Config):

  def __init__(self):
    super(ResNet32DNConfig, self).__init__()
    self.model = "resnet-32-dn"
    self.sum_window = [[7, 7], [5, 5], [5, 5], [5, 5]]
    self.sup_window = [[7, 7], [5, 5], [5, 5], [5, 5]]
    self.norm_field = "div"
    self.norm_affine = False
    self.learn_sigma = True
    self.sigma_init = 0.5


class ResNet32LNConfig(ResNet32Config):

  def __init__(self):
    super(ResNet32LNConfig, self).__init__()
    self.model = "resnet-32-ln"
    self.norm_field = "layer"
    self.norm_affine = False
    self.sigma_init = 0.5


class ResNet110Config(ResNet32Config):

  def __init__(self):
    super(ResNet110Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-110
    self.model = "resnet-110"


class ResNet164Config(ResNet32Config):

  def __init__(self):
    super(ResNet164Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-164
    self.use_bottleneck = True
    self.model = "resnet-164"
