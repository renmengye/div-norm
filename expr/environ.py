from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import numpy as np
import datetime


def gen_id(prefix):
  return "{}_{}-{:03d}".format(
      prefix,
      datetime.datetime.now().isoformat(chr(ord("-"))).replace(
          ":", "-").replace(".", "-"),
      int(np.random.rand() * 1000))


def get_device(gpu):
  if gpu == -1:
    return "/cpu:0"
  else:
    return "/gpu:{:d}".format(gpu)


class BasicEnvironment(object):

  def __init__(self,
               device="/gpu:0",
               gpu=0,
               num_gpu=1,
               machine=None,
               num_cpu=2,
               dataset="cifar-10",
               data_folder="cifar-10",
               logs_folder="logs/default",
               save_folder="results",
               run_validation=True,
               verbose=True,
               exp_id="exp_cifar",
               description=None,
               valid_num_fold=10,
               valid_fold_id=0):
    self.run_validation = run_validation
    self.dataset = dataset
    self.device = device
    self.gpu = gpu
    self.num_gpu = num_gpu
    self.machine = machine
    self.num_cpu = num_cpu
    self.data_folder = data_folder
    self.exp_id = exp_id
    self.verbose = verbose
    self.logs_folder = logs_folder
    self.save_folder = save_folder
    self.description = description
    self.valid_num_fold = valid_num_fold
    self.valid_fold_id = valid_fold_id

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)

  @classmethod
  def from_json(cls, s):
    dic = json.loads(s)
    config = cls()
    config.__dict__ = dic
    return config
