#!/usr/bin/python
"""
Authors: Mengye Ren (mren@cs.toronto.edu) Renjie Liao (rjliao@cs.toronto.edu)

The following code explores different normalization schemes for super-resolution.

Usage:
python run_sr_exp.py       --model           [MODEL NAME]        \
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
         2) dnms
         3) dn-star
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
import cv2
import sys
import tensorflow as tf

from data import SRHDF5MatlabDataset
from expr import BasicEnvironment, gen_id, get_device
from utils import BatchIterator, ConcurrentBatchIterator
from utils import logger
from utils import progress_bar as pb
from utils import img_utils as ut

import sr_exp_config as conf
from cnn_model import CNNModelSR
from run_cifar_exp import get_environ, get_iter

log = logger.get()

FLAGS = tf.app.flags.FLAGS


def get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return conf.BaselineConfig.from_json(open(FLAGS.config, "r").read())
  else:
    return conf.get_config(FLAGS.dataset, FLAGS.model)


def get_dataset(name,
                folder,
                split,
                num_patch_per_img=100,
                patch_size=33,
                stride=1,
                resize_factor=3):
  """Gets CIFAR datasets.

  Args:
      name: "cifar-10" or "cifar-100".
      folder: Dataset folder.
      split: "train", "traintrain", "trainval", or "test".

  Returns:
      dp: Dataset object.
  """
  if name == "HDF5Matlab":
    dp = SRHDF5MatlabDataset(
        split,
        folder=folder,
        num_patch=num_patch_per_img,
        patch_size=patch_size,
        stride=stride,
        resize_factor=resize_factor)
  else:
    raise Exception("Unknown dataset {}".format(name))
  return dp


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
      f.write("train_mse_loss.csv,csv,Train Loss (MSE)\n")
      f.write("valid_ssim.csv,csv,Valid SSIM\n")
      f.write("valid_psnr.csv,csv,Valid PSNR\n")

    self.train_file_name = os.path.join(logs_folder, "train_mse_loss.csv")
    if not os.path.exists(self.train_file_name):
      with open(self.train_file_name, "w") as f:
        f.write("step,time,Training MSE\n")

    self.valid_psnr_file_name = os.path.join(logs_folder, "valid_psnr.csv")
    if not os.path.exists(self.valid_psnr_file_name):
      with open(self.valid_psnr_file_name, "w") as f:
        f.write("step,time,PSNR\n")

    self.valid_ssim_file_name = os.path.join(logs_folder, "valid_ssim.csv")
    if not os.path.exists(self.valid_ssim_file_name):
      with open(self.valid_ssim_file_name, "w") as f:
        f.write("step,time,SSIM\n")

  def log_train_loss(self, niter, loss):
    """Writes training L2 loss."""
    log.info("Train Step = {:06d} || L2 loss = {:.4e}".format(niter + 1, loss))
    with open(self.train_file_name, "a") as f:
      f.write("{:d},{:s},{:e}\n".format(
          niter + 1, datetime.datetime.now().isoformat(), loss))

  def log_valid_psnr(self, niter, psnr):
    """Writes validation psnr."""
    log.info("Average PSNR = {:.3f}".format(psnr))
    with open(self.valid_psnr_file_name, "a") as f:
      f.write("{:d},{:s},{:.5f}\n".format(
          niter + 1, datetime.datetime.now().isoformat(), psnr))

  def log_valid_ssim(self, niter, ssim):
    """Writes validation ssim."""
    log.info("Average PSNR = {:.3f}".format(ssim))
    with open(self.valid_ssim_file_name, "a") as f:
      f.write("{:d},{:s},{:.5f}\n".format(
          niter + 1, datetime.datetime.now().isoformat(), ssim))


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
          m = CNNModelSR(config, is_training=True)

    with tf.name_scope("Valid"):  # also include testing in this graph
      with tf.variable_scope("Model", reuse=True):
        with tf.device(environ.device):
          mvalid = CNNModelSR(config, is_training=False)

    # Initializes variables.
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())

      def train_step():
        """Train step."""
        batch = train_iter.next()

        feed_data = {
            m.input:
            np.expand_dims(batch["img"], axis=3),
            m.label:
            np.expand_dims(
                ut.crop_img(batch["label"], config.crop_border), axis=3)
        }

        cost, l2_loss, _ = sess.run(
            [m.cost, m.l2_loss, m.train_op], feed_dict=feed_data)
        return l2_loss

      def evaluate(data_iter, nbatches):
        """Runs evaluation."""
        count = 0
        PSNR = 0.0
        SSIM = 0.0
        crop_border = config.crop_border

        if nbatches == -1:
          iter_ = data_iter
        else:
          iter_ = range(nbatches)

        for bb in iter_:
          if nbatches == -1:
            batch = bb
          else:
            batch = data_iter.next()

          # deal with gray images
          is_rgb_img = False if len(batch["img"].shape) < 3 else True

          if not is_rgb_img:
            img_y = batch["img"]
            label_y = batch["label"]
          else:
            # note Matlab format is Ycbcr
            img_y = batch["img"][:, :, 0]
            img_cb = batch["img"][:, :, 1]
            img_cr = batch["img"][:, :, 2]
            label_y = batch["label"][:, :, 0]

          label_y = ut.crop_img(label_y, crop_border)
          feed_data = {
              mvalid.input:
              np.expand_dims(np.expand_dims(img_y, axis=0), axis=3)
          }

          output_img = sess.run(mvalid.output, feed_dict=feed_data)
          output_img = ut.clip_img(np.squeeze(output_img *
                                              255.0))  # clip pixel value
          PSNR += ut.compute_psnr(output_img, label_y)
          SSIM += ut.compute_ssim(output_img, label_y)

          if not is_rgb_img:
            save_input_img = ut.crop_img(
                ut.post_process(img_y * 255.0), crop_border)
            save_output_img = ut.post_process(output_img)
          else:
            save_input_img = np.zeros_like(batch["img"])
            # note OpenCV format is Ycrcb
            save_input_img[:, :, 0] = ut.clip_img(img_y * 255.0)
            save_input_img[:, :, 1] = img_cr
            save_input_img[:, :, 2] = img_cb

            save_input_img = ut.crop_img(
                ut.post_process(
                    cv2.cvtColor(
                        save_input_img.astype(np.uint8), cv2.COLOR_YCR_CB2BGR)),
                crop_border)

            save_output_img = np.zeros_like(save_input_img)
            save_output_img[:, :, 0] = output_img
            save_output_img[:, :, 1] = img_cr[crop_border:-crop_border,
                                              crop_border:-crop_border]
            save_output_img[:, :, 2] = img_cb[crop_border:-crop_border,
                                              crop_border:-crop_border]

            save_output_img = ut.post_process(
                cv2.cvtColor(
                    save_output_img.astype(np.uint8), cv2.COLOR_YCR_CB2BGR))

          cv2.imwrite(
              os.path.join(save_folder,
                           "test_input_{:05d}.png".format(count + 1)),
              save_input_img)

          cv2.imwrite(
              os.path.join(save_folder,
                           "test_output_{:05d}.png".format(count + 1)),
              save_output_img)

          count += 1

        PSNR /= count
        SSIM /= count

        return PSNR, SSIM

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
          l2_loss = train_step()
          if (niter + 1) % config.disp_iter == 0 or niter == 0:
            exp_logger.log_train_loss(niter, l2_loss)
          if (niter + 1) % config.valid_iter == 0 or niter == 0:
            log.info("Experment ID {}".format(environ.exp_id))
            test_iter.reset()
            psnr, ssim = evaluate(test_iter, -1)
            exp_logger.log_valid_psnr(niter, psnr)
            exp_logger.log_valid_ssim(niter, ssim)
          if (niter + 1) % config.save_iter == 0:
            save()
        test_iter.reset()
        psnr, ssim = evaluate(test_iter, -1)
        return psnr, ssim

      psnr, ssim = train()
  return psnr, ssim


def main():
  # Loads parammeters.
  config = get_config()
  FLAGS.dataset = config.dataset
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
      num_patch_per_img=config.num_patch_per_img,
      patch_size=config.patch_size,
      resize_factor=int(config.resize_factor))

  test_data = get_dataset(
      environ.dataset,
      environ.data_folder,
      test_str,
      num_patch_per_img=config.num_patch_per_img,
      patch_size=config.patch_size,
      resize_factor=int(config.resize_factor))

  # Trains a model.
  psnr, ssim = train_model(config, environ, train_data, test_data)
  log.info("Final test PSNR/SSIM = {:.5f}/{:.5f}".format(psnr, ssim))


if __name__ == "__main__":
  main()
