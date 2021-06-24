# Copyright 2021 The XMC-GAN Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from absl import logging

import jax
from jax import numpy as jnp
from PIL import Image


def make_grid(samples, show_num=64):
  """Save image to image grid."""
  batch_size, height, width, c = samples.shape
  if batch_size < show_num:
    logging.info("show_num is cut by the small batch size to %d", batch_size)
    show_num = batch_size
  h_num = int(math.sqrt(show_num))
  w_num = int(show_num / h_num)
  grid_num = h_num * w_num

  samples = samples[0:grid_num]
  samples = samples.reshape(h_num, w_num, height, width, c)
  samples = samples.swapaxes(1, 2)
  samples = samples.reshape(height * h_num, width * w_num, c)
  # samples = np.array(samples)
  return samples


def save_image(ndarray,
               fp,
               nrow=8,
               padding=2,
               pad_value=0.0,
               image_format=None):
  """Make a grid of images and Save it into an image file.

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp: A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    image_format(Optional):  If omitted, the format to use is determined from
      the filename extension. If a file object was used instead of a filename,
      this parameter should always be used.
  """
  if not (isinstance(ndarray, jnp.ndarray) or
          (isinstance(ndarray, list) and
           all(isinstance(t, jnp.ndarray) for t in ndarray))):
    raise TypeError("array_like of tensors expected, got {}".format(
        type(ndarray)))

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                       padding)
  num_channels = ndarray.shape[3]
  grid = jnp.full(
      (height * ymaps + padding, width * xmaps + padding, num_channels),
      pad_value).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = jax.ops.index_update(
          grid, jax.ops.index[y * height + padding:(y + 1) * height,
                              x * width + padding:(x + 1) * width], ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
  im = Image.fromarray(ndarr.copy())
  im.save(fp, format=image_format)
