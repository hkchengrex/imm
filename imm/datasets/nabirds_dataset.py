# ==========================================================
# Author: Tomas Jakab
# ==========================================================
from __future__ import division

import os
import numpy as np
import tensorflow as tf

from imm.datasets.tps_dataset import TPSDataset



def load_dataset(data_root, dataset, subset):

    image_dir = os.path.join(data_root, subset)

    images = os.listdir(image_dir)

    return image_dir, images, None



class NABirdsdataset(TPSDataset):
  LANDMARK_LABELS = {'left_eye': 0, 'right_eye': 1}
  N_LANDMARKS = 5


  def __init__(self, data_dir, subset, dataset=None, max_samples=None,
               image_size=[128, 128], order_stream=False, landmarks=False,
               tps=True, vertical_points=10, horizontal_points=10,
               rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
               warpsd=[0.001, 0.005, 0.001, 0.01],
               name='CelebADataset'):

    super(NABirdsdataset, self).__init__(
        data_dir, subset, max_samples=max_samples,
        image_size=image_size, order_stream=order_stream, landmarks=landmarks,
        tps=tps, vertical_points=vertical_points,
        horizontal_points=horizontal_points, rotsd=rotsd, scalesd=scalesd,
        transsd=transsd, warpsd=warpsd, name=name)

    assert dataset is not None

    self._dataset = dataset

    self._image_dir, self._images, self._keypoints = load_dataset(
        self._data_dir, self._dataset, self._subset)

    # Force override, no information stored here
    self._keypoints = np.zeros((len(self._images), 5, 2))


  def _get_sample_dtype(self):
    d =  {'image': tf.string,
          'landmarks': tf.float32}
    d.update({k: tf.int32 for k in self.LANDMARK_LABELS.keys()})
    return d


  def _get_sample_shape(self):
    d = {'image': None,
         'landmarks': [self.N_LANDMARKS, 2]}
    d.update({k: [] for k in self.LANDMARK_LABELS.keys()})
    return d


  def _proc_im_pair(self, inputs):
    with tf.name_scope('proc_im_pair'):
      height, width = self._image_size[:2]

      # read in the images:
      image = self._read_image_tensor_or_string(inputs['image'])

      if 'landmarks' in inputs:
        landmarks = inputs['landmarks']
      else:
        landmarks = None

      crop_percent = 0.8
      assert self._image_size[0] == self._image_size[1]
      final_sz = self._image_size[0]
      resize_sz = np.round(final_sz / crop_percent).astype(np.int32)
      margin = np.round((resize_sz - final_sz) / 2.0).astype(np.int32)

      if landmarks is not None:
        original_sz = tf.shape(image)[:2]
        landmarks = self._resize_points(
            landmarks, original_sz, [resize_sz, resize_sz])
        landmarks -= margin

      image = tf.image.resize_images(image, [resize_sz, resize_sz],
          tf.image.ResizeMethod.BILINEAR, align_corners=True)
      # take central crop
      image = image[margin:margin + final_sz, margin:margin + final_sz]

      mask = self._get_smooth_mask(height, width, 10, 20)[:, :, None]

      future_landmarks = landmarks
      future_image = image

      inputs = {k: inputs[k] for k in self._get_sample_dtype().keys()}
      inputs.update({'image': image, 'future_image': future_image,
                     'mask': mask, 'landmarks': landmarks,
                     'future_landmarks': future_landmarks})
    return inputs
