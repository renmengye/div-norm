#!/usr/bin/python
"""
Authors: Mengye Ren (mren@cs.toronto.edu) Renjie Liao (rjliao@cs.toronto.edu)

The following code explores different normalization schemes in CNN on CIFAR
datasets.

Usage:
python run_cifar_exp.py    --model           [MODEL NAME]        \
                           --config          [CONFIG FILE]       \
                           --env             [ENV FILE]          \
                           --dataset         [DATASET]           \
                           --data_folder     [DATASET FOLDER]    \
                           --validation                          \
                           --no_validation
                           --logs            [LOGS FOLDER]       \
                           --results         [SAVE FOLDER]       \
                           --gpu             [GPU ID]

Flags:
    --model: Model type. Available options are:
         1) base
         2) base-wd
         3) base-drop
         4) base-wd-drop
         5) bn
         6) bn-s
         7) bn-l1
         8) bn-star
         9) ln
        10) ln-s
        11) ln-l1
        12) ln-star
        13) dn
        14) dn-star
    --config: Not using the pre-defined configs above, specify the JSON file
    that contains model configurations.
    --dataset: Dataset name. Available options are: 1) cifar-10 2) cifar-100.
    --data_folder: Path to data folder, default is {DATASET}.
    --validation: Evaluating experiments on validation set.
    --no_validation: Evaluating experiments on test set.
    --logs: Path to logs folder, default is logs/default.
    --results: Path to save folder, default is results.
    --gpu: Which GPU to run, default is 0, -1 for CPU.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import json
import numpy as np
import os
import sys
import tensorflow as tf

from data import CIFAR10Dataset, CIFAR100Dataset
from expr import BasicEnvironment, gen_id, get_device
from utils import BatchIterator, ConcurrentBatchIterator
from utils import logger
from utils import progress_bar as pb

import cifar_exp_config as conf
from cnn_model import CNNModel
from resnet_model import ResNetModel

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "manually defined config file")
flags.DEFINE_string("dataset", "cifar-10", "dataset name")
flags.DEFINE_string("data_folder", None, "data folder")
flags.DEFINE_string("description", None, "description")
flags.DEFINE_string("env", None, "manually set environ file")
flags.DEFINE_integer("gpu", 0, "GPU ID")
flags.DEFINE_string("id", None, "experiment ID")
flags.DEFINE_string("results", "../results/cifar", "saving folder")
flags.DEFINE_string("logs", "../logs/default", "logging folder")
flags.DEFINE_string("model", "base", "model type")
flags.DEFINE_bool("validation", False, "whether use validation set")
flags.DEFINE_integer("valid_fold_id", 0, "cross validation split ID")
flags.DEFINE_integer("valid_num_fold", 10, "k-fold cross validation")
flags.DEFINE_bool("verbose", False, "verbose logging")
FLAGS = flags.FLAGS


def get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return conf.BaselineConfig.from_json(open(FLAGS.config, "r").read())
  else:
    return conf.get_config(FLAGS.dataset, FLAGS.model)


def get_environ():
  """Gets an environment object."""
  # Manually set environment.
  if FLAGS.env is not None:
    return BasicEnvironment.from_json(open(FLAGS.env, "r").read())

  if FLAGS.data_folder is None:
    data_folder = FLAGS.dataset
  else:
    data_folder = FLAGS.data_folder
  exp_id = "exp_" + FLAGS.dataset + "_" + FLAGS.model
  if FLAGS.id is None:
    exp_id = gen_id(exp_id)
  else:
    exp_id = FLAGS.id
  return BasicEnvironment(
      device=get_device(FLAGS.gpu),
      dataset=FLAGS.dataset,
      data_folder=data_folder,
      logs_folder=FLAGS.logs,
      save_folder=FLAGS.results,
      run_validation=FLAGS.validation,
      verbose=FLAGS.verbose,
      exp_id=exp_id,
      description=FLAGS.description,
      valid_num_fold=FLAGS.valid_num_fold,
      valid_fold_id=FLAGS.valid_fold_id)


def get_dataset(name,
                folder,
                split,
                num_fold=10,
                fold_id=0,
                data_aug=False,
                whiten=False,
                div255=False):
  """Gets CIFAR datasets.

  Args:
      name: "cifar-10" or "cifar-100".
      folder: Dataset folder.
      split: "train", "traintrain", "trainval", or "test".

  Returns:
      dp: Dataset object.
  """
  if name == "cifar-10":
    dp = CIFAR10Dataset(
        folder,
        split,
        num_fold=num_fold,
        fold_id=fold_id,
        data_aug=data_aug,
        whiten=whiten,
        div255=div255)
  elif name == "cifar-100":
    dp = CIFAR100Dataset(
        folder,
        split,
        num_fold=num_fold,
        fold_id=fold_id,
        data_aug=data_aug,
        whiten=whiten,
        div255=div255)
  else:
    raise Exception("Unknown dataset {}".format(dataset))
  return dp


def get_iter(dataset,
             batch_size=100,
             shuffle=False,
             cycle=False,
             log_epoch=-1,
             seed=0,
             prefetch=False,
             num_worker=20,
             queue_size=300):
  """Gets a data iterator.

  Args:
      dataset: Dataset object.
      batch_size: Mini-batch size.
      shuffle: Whether to shuffle the data.
      cycle: Whether to stop after one full epoch.
      log_epoch: Log progress after how many iterations.

  Returns:
      b: Batch iterator object.
  """
  b = BatchIterator(
      dataset.get_size(),
      batch_size=batch_size,
      shuffle=shuffle,
      cycle=cycle,
      get_fn=dataset.get_batch_idx,
      log_epoch=log_epoch,
      seed=seed)
  if prefetch:
    b = ConcurrentBatchIterator(
        b, max_queue_size=queue_size, num_threads=num_worker, log_queue=-1)
  return b


class ExperimentLogger():
  """Writes experimental logs to CSV file."""

  def __init__(self, logs_folder):
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

    with open(catalog_file, "a") as f:
      f.write("train_ce.csv,csv,Train Loss (Cross Entropy)\n")
      f.write("train_acc.csv,csv,Train Accuracy\n")
      f.write("valid_acc.csv,csv,Validation Accuracy\n")

    self.train_file_name = os.path.join(logs_folder, "train_ce.csv")
    if not os.path.exists(self.train_file_name):
      with open(self.train_file_name, "w") as f:
        f.write("step,time,ce\n")

    self.trainval_file_name = os.path.join(logs_folder, "train_acc.csv")
    if not os.path.exists(self.trainval_file_name):
      with open(self.trainval_file_name, "w") as f:
        f.write("step,time,acc\n")

    self.val_file_name = os.path.join(logs_folder, "valid_acc.csv")
    if not os.path.exists(self.val_file_name):
      with open(self.val_file_name, "w") as f:
        f.write("step,time,acc\n")

  def log_train_ce(self, niter, ce):
    """Writes training CE."""
    log.info("Train Step = {:06d} || CE loss = {:.4e}".format(niter + 1, ce))
    with open(self.train_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          niter + 1, datetime.datetime.now().isoformat(), ce))

  def log_train_acc(self, niter, acc):
    """Writes training accuracy."""
    log.info("Train accuracy = {:.3f}".format(acc * 100))
    with open(self.trainval_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          niter + 1, datetime.datetime.now().isoformat(), acc))

  def log_valid_acc(self, niter, acc):
    """Writes validation accuracy."""
    log.info("Valid accuracy = {:.3f}".format(acc * 100))
    with open(self.val_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          niter + 1, datetime.datetime.now().isoformat(), acc))


def train_model(config, environ, train_data, test_data, trainval_data=None):
  """Trains a CIFAR model.

  Args:
      config: Config object
      environ: Environ object
      train_data: Dataset object
      test_data: Dataset object

  Returns:
      acc: Final test accuracy
  """
  np.random.seed(0)
  if not hasattr(config, "seed"):
    tf.set_random_seed(1234)
    log.info("Setting tensorflow random seed={:d}".format(1234))
  else:
    log.info("Setting tensorflow random seed={:d}".format(config.seed))
    tf.set_random_seed(config.seed)
  if environ.verbose:
    verbose_level = 0
  else:
    verbose_level = 2

  if trainval_data is None:
    trainval_data = train_data

  log.info("Environment: {}".format(environ.__dict__))
  log.info("Config: {}".format(config.__dict__))

  save_folder = os.path.join(environ.save_folder, environ.exp_id)
  logs_folder = os.path.join(environ.logs_folder, environ.exp_id)
  with log.verbose_level(verbose_level):
    exp_logger = ExperimentLogger(logs_folder)

    if not hasattr(config, "seed"):
      data_seed = 0
    else:
      data_seed = config.seed

    # Gets data iterators.
    train_iter = get_iter(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        cycle=True,
        prefetch=config.prefetch,
        seed=data_seed,
        num_worker=25,
        queue_size=500)
    trainval_iter = get_iter(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        cycle=True,
        prefetch=config.prefetch,
        num_worker=10,
        queue_size=200)
    test_iter = get_iter(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        cycle=False,
        prefetch=config.prefetch,
        num_worker=10,
        queue_size=200)

    # Builds models.
    log.info("Building models")
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None):
        with tf.device(environ.device):
          if config.model.startswith("resnet"):
            m = ResNetModel(config, is_training=True)
          else:
            m = CNNModel(config, is_training=True)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True):
        with tf.device(environ.device):
          if config.model.startswith("resnet"):
            mvalid = ResNetModel(config, is_training=False)
          else:
            mvalid = CNNModel(config, is_training=False)

    # Initializes variables.
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initilaizer())

      def train_step():
        """Train step."""
        batch = train_iter.next()
        feed_data = {m.input: batch["img"], m.label: batch["label"]}
        cost, ce, _ = sess.run([m.cost, m.cross_ent, m.train_op],
                               feed_dict=feed_data)
        return ce

      def evaluate(data_iter, nbatches):
        """Runs evaluation."""
        num_correct = 0.0
        count = 0
        if nbatches == -1:
          iter_ = data_iter
        else:
          iter_ = range(nbatches)

        for bb in iter_:
          if nbatches == -1:
            batch = bb
          else:
            batch = data_iter.next()
          feed_data = {mvalid.input: batch["img"]}
          y = sess.run(mvalid.output, feed_dict=feed_data)
          pred_label = np.argmax(y, axis=1)
          num_correct += np.sum(
              np.equal(pred_label, batch["label"]).astype(float))
          count += pred_label.size
        acc = (num_correct / count)
        return acc

      def save():
        """Snapshots a model."""
        if not os.path.isdir(save_folder):
          os.makedirs(save_folder)
          config_file = os.path.join(save_folder, "conf.json")
          environ_file = os.path.join(save_folder, "env.json")
          with open(config_file, "w") as f:
            f.write(config.to_json())
          with open(environ_file, "w") as f:
            f.write(environ.to_json())
        log.info("Saving to {}".format(save_folder))
        saver.save(
            sess,
            os.path.join(save_folder, "model.ckpt"),
            global_step=m.global_step)

      def train():
        """Train loop."""
        lr = config.base_learn_rate
        lr_decay_steps = config.lr_decay_steps
        max_train_iter = config.max_train_iter
        m.assign_lr(sess, lr)

        if environ.verbose:
          loop = range(max_train_iter)
        else:
          loop = pb.get(max_train_iter)

        for niter in loop:
          # decrease learning rate
          if len(lr_decay_steps) > 0:
            if (niter + 1) == lr_decay_steps[0]:
              lr *= 0.1
              m.assign_lr(sess, lr)
              lr_decay_steps.pop(0)
          ce = train_step()
          if (niter + 1) % config.disp_iter == 0 or niter == 0:
            exp_logger.log_train_ce(niter, ce)
          if (niter + 1) % config.valid_iter == 0 or niter == 0:
            acc = evaluate(trainval_iter, 10)
            exp_logger.log_train_acc(niter, acc)
            test_iter.reset()
            acc = evaluate(test_iter, -1)
            log.info("Experment ID {}".format(environ.exp_id))
            exp_logger.log_valid_acc(niter, acc)
          if (niter + 1) % config.save_iter == 0:
            save()
        test_iter.reset()
        acc = evaluate(test_iter, -1)
        return acc

      acc = train()
  return acc


def main():
  # Loads parammeters.
  config = get_config()
  environ = get_environ()
  if environ.run_validation:
    train_str = "traintrain"
    test_str = "trainval"
    log.warning("Running validation set")
  else:
    train_str = "train"
    test_str = "test"

  # Configures dataset objects.
  log.info("Building dataset")
  train_data = get_dataset(
      environ.dataset,
      environ.data_folder,
      train_str,
      num_fold=environ.valid_num_fold,
      fold_id=environ.valid_fold_id,
      data_aug=config.data_aug,
      whiten=config.whiten,
      div255=config.div255)
  if config.data_aug:
    trainval_data = get_dataset(
        environ.dataset,
        environ.data_folder,
        train_str,
        num_fold=environ.valid_num_fold,
        fold_id=environ.valid_fold_id,
        data_aug=False,
        whiten=config.whiten,
        div255=config.div255)
  else:
    trainval_data = train_data
  test_data = get_dataset(
      environ.dataset,
      environ.data_folder,
      test_str,
      num_fold=environ.valid_num_fold,
      fold_id=environ.valid_fold_id,
      data_aug=False,
      whiten=config.whiten,
      div255=config.div255)

  # Trains a model.
  acc = train_model(config, environ, train_data, test_data, trainval_data)
  log.info("Final test accuracy = {:.3f}".format(acc * 100))


if __name__ == "__main__":
  main()
