from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import json


def get_config(model):
  config = BaselineConfig()
  model_cell = model.split("-")[-1]
  if model_cell == "lstm":
    config.rnn_cell = "lstm"
  else:
    config.rnn_cell = "rnn"
    act_fn = model.split("-")[-2]
    config.state_act_fn = act_fn

  # Modify normalization methods.
  norm_field = model.split("-")[0]

  # Baseline.
  if norm_field == "base":
    config.norm_field = "no"

  # Batch normalization.
  elif norm_field == "bn":
    config.norm_field = "batch"
    config.norm_activation = False
    config.norm_affine = True

    # Sigma value.
    if model.startswith("bn-s") or model.startswith("bn-star"):
      config.sigma_init = 1e0
    else:
      config.sigma_init = 1e-2

    # L1 regularization value.
    if model_cell == "lstm":
      if "l1" in model:
        config.l1_reg = 1e-2
      elif "star" in model:
        config.l1_reg = 1e-1
    else:
      if act_fn == "tanh":
        if "l1" in model:
          config.l1_reg = 1e-1
        elif "star" in model:
          config.l1_reg = 1e-4
      elif act_fn == "relu":
        if "l1" in model:
          config.l1_reg = 1e-4
        elif "star" in model:
          config.l1_reg = 1e-1
      else:
        raise ValueError("Unknown activation function: {}".format(act_fn))

  # Layer normalization.
  elif norm_field == "ln":
    config.norm_field = "layer"

    # Layer norm technical details.
    if model.startswith("ln-s") or model.startswith(
        "ln-l1") or model.startswith("ln-star"):
      config.norm_activation = True
      config.norm_affine = False
    else:
      # Original version of the layer normalization.
      config.norm_activation = False
      config.norm_affine = True

    # Sigma value.
    if model.startswith("ln-s") or model.startswith("ln-star"):
      config.sigma_init = 1e0
    else:
      config.sigma_init = 1e-2

    # L1 regularization value.
    if model_cell == "lstm":
      if "l1" in model:
        config.l1_reg = 1e-2
      elif "star" in model:
        config.l1_reg = 1e-3
    else:
      config.l1_reg = 1e-2

  # Divisive normalization.
  elif norm_field == "dn":
    config.norm_field = "div"
    config.norm_activation = True
    config.norm_affine = False

    # Sigma value.
    if model.startswith("dn-s") or model.startswith("dn-star"):
      config.sigma_init = 1e0
    else:
      config.sigma_init = 1e-2

    # Radius of the normalization neighbourhood.
    if model_cell == "lstm":
      config.dn_window = 30
    else:
      config.dn_window = 60

    # L1 regularization value.
    if "star" in model:
      if model_cell == "lstm":
        config.l1_reg = 1e-4
      else:
        config.l1_reg = 1e-2

  else:
    raise ValueError("Unknown norm field: {}".format(norm_field))

  # Modify learning rate.
  if (model.startswith("base-tanh") or model.startswith("base-relu") or
      model.startswith("bn-relu")):
    config.learning_rate = 0.1
  else:
    config.learning_rate = 1.0

  # Model name.
  config.model = model

  return config


class BaselineConfig(object):
  """Baseline LSTM config."""

  def __init__(self):
    self.init_scale = 0.1
    self.learning_rate = 1.0
    self.max_grad_norm = 5
    self.num_layers = 2
    self.num_steps = 20
    self.hidden_size = 200
    self.max_epoch = 4
    self.max_max_epoch = 13
    self.keep_prob = 1.0
    self.lr_decay = 0.5
    self.batch_size = 20
    self.vocab_size = 10000
    self.level = "word"
    self.optimizer = "gd"
    self.rnn_cell = "lstm"
    self.norm_field = "no"
    self.state_act_fn = "tanh"
    self.gate_act_fn = "sigmoid"
    self.l1_reg = 0.0
    self.model = "base-lstm"
    self.sigma_init = 1e-2
    self.norm_affine = False
    self.dn_window = 0

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)

  @classmethod
  def from_json(cls, s):
    dic = json.loads(s)
    config = cls()
    config.__dict__ = dic
    return config
