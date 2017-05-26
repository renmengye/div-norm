import numpy as np
import SSIM_index as SSIM


def compute_ssim(pred_img, gt_img):
  return SSIM.compute_ssim(pred_img, gt_img)


def compute_psnr(pred_img, gt_img):

  diff = pred_img.astype(np.float32) - gt_img.astype(np.float32)
  MSE = np.mean(np.power(diff, 2.0))

  # PSNR = 20 * np.log10(255.0) - 10 * np.log10(MSE)
  PSNR = 10.0 * np.log10(255.0 * 255.0 / MSE)

  return PSNR


def crop_img(img, crop_size):
  if len(img.shape) == 4:
    return img[:, crop_size:-crop_size, crop_size:-crop_size, :]
  elif len(img.shape) == 3:
    if img.shape[2] == 1 or img.shape[2] == 3:
      return img[crop_size:-crop_size, crop_size:-crop_size, :]
    else:
      return img[:, crop_size:-crop_size, crop_size:-crop_size]
  elif len(img.shape) == 2:
    return img[crop_size:-crop_size, crop_size:-crop_size]
  else:
    raise ValueError('Input image error!')


def clip_img(img):
  img[img > 255.0] = 255.0
  img[img < 0.0] = 0.0
  return img


def post_process(img):
  return clip_img(img).astype(np.uint8)
  # return (img * 255.0).astype(np.uint8)