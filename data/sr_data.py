import os
import h5py
import numpy as np

from utils import logger

log = logger.get()


def random_crop(hr_img, lr_img, num_per_img, patch_size, stride):

  hr_patch = []
  lr_patch = []
  height = hr_img.shape[0]
  width = hr_img.shape[1]

  if height != lr_img.shape[0] or width != lr_img.shape[1]:
    raise ValueError("low-res and high-res images should be of the same size!")

  if len(lr_img.shape) > 3 or len(lr_img.shape) < 2 or len(
      hr_img.shape) > 3 or len(hr_img.shape) < 2:
    raise ValueError("low-res and high-res images are of wrong shapes!")

  idx_h, idx_w = np.meshgrid(
      np.arange(0, height - patch_size, stride),
      np.arange(0, width - patch_size, stride))

  idx_h = idx_h.flatten()
  idx_w = idx_w.flatten()

  rand_idx = np.random.permutation(np.arange(len(idx_w)))

  if len(rand_idx) < num_per_img:
    return [], []

  rand_idx = rand_idx[:num_per_img]

  for ii in xrange(num_per_img):
    hh = idx_h[rand_idx[ii]]
    ww = idx_w[rand_idx[ii]]

    if len(hr_img.shape) == 3:
      hr_patch += [hr_img[hh:hh + patch_size, ww:ww + patch_size, :]]
      lr_patch += [lr_img[hh:hh + patch_size, ww:ww + patch_size, :]]
    else:
      hr_patch += [hr_img[hh:hh + patch_size, ww:ww + patch_size]]
      lr_patch += [lr_img[hh:hh + patch_size, ww:ww + patch_size]]

  return hr_patch, lr_patch


class SRHDF5MatlabDataset():

  def __init__(self, split, folder, num_patch=100, patch_size=33, stride=1):
    self.split = split
    self.num_patch = num_patch
    self.patch_size = patch_size
    self.stride = stride

    self.h5_dict = h5py.File(os.path.join(folder, "data.h5"), "r")
    self.train_hr_imgs = self.h5_dict["hr_img"]
    self.train_lr_imgs = self.h5_dict["lr_img"]
    self.test_hr_imgs = self.h5_dict["hr_img_test"]
    self.test_lr_imgs = self.h5_dict["lr_img_test"]
    self.train_hr_img_names = sorted(self.train_hr_imgs.keys())
    self.train_lr_img_names = sorted(self.train_lr_imgs.keys())
    self.test_hr_img_names = sorted(self.test_hr_imgs.keys())
    self.test_lr_img_names = sorted(self.test_lr_imgs.keys())
    self.num_train_imgs = len(self.train_hr_img_names)
    self.num_test_imgs = len(self.test_hr_img_names)

    log.info("Number of training images = {}".format(self.num_train_imgs))
    log.info("Number of testing images = {}".format(self.num_test_imgs))

  def get_size(self):
    if self.split == "train":
      return self.num_train_imgs
    else:
      return self.num_test_imgs

  def get_batch_idx(self, idx):
    if self.split == "train":
      train_hr_patch = []
      train_lr_patch = []

      count = 0
      cur_idx = idx[count]

      while count < len(idx):
        hr_img = np.transpose(
            self.train_hr_imgs[self.train_hr_img_names[cur_idx]][:])
        lr_img = np.transpose(
            self.train_lr_imgs[self.train_lr_img_names[cur_idx]][:])

        hr_patch, lr_patch = random_crop(hr_img, lr_img, self.num_patch,
                                         self.patch_size, self.stride)

        if hr_patch and lr_patch:
          train_hr_patch += [np.concatenate(np.expand_dims(hr_patch, axis=0))]
          train_lr_patch += [np.concatenate(np.expand_dims(lr_patch, axis=0))]

          count += 1
          if count < len(idx):
            cur_idx = idx[count]
        else:
          cur_idx = (cur_idx + 1) % self.num_imgs

      return {
          "img": np.concatenate(train_lr_patch, axis=0),
          "label": np.concatenate(train_hr_patch, axis=0)
      }
    else:
      assert len(idx) == 1

      return {
          "img":
          np.transpose(self.test_lr_imgs[self.test_lr_img_names[idx[0]]]),
          "label":
          np.transpose(self.test_hr_imgs[self.test_hr_img_names[idx[0]]])
      }
