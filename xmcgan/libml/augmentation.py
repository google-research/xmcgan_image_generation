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

from typing import Union
import gin
import jax
import jax.numpy as jnp
import tensorflow as tf

_SHIFT = "shift"
_ZOOM_CROP = "zoom_crop"


@gin.configurable()
def augment(
    x: tf.Tensor,
    method: str = _SHIFT,
    random_flip: bool = True,
    resize_method: str = "bilinear",
    seed: Union[None, jnp.ndarray, tf.Tensor] = None,
    **kwargs,
) -> tf.Tensor:
  """Randomly augments the input image.

  Args:
    x: Input image tensor.
    method: Augmentation method. Supported methods are "shift" and "zoom_crop".
    random_flip: Whether to random flip the image.
    resize_method: Image resize method, 'bilinear' or 'nearest'.
    seed: Random seed for augmentation.
    **kwargs: Additional input arguments.

  Returns:
    The augmented image tensor.

  Raises:
    NotImplementedError: An error occurred when augmentation method is not
      supported.
  """
  rngs = [None, None, None]
  rng_available = seed is not None
  if rng_available:
    if isinstance(seed, tf.Tensor):
      rngs = tf.unstack(tf.random.experimental.stateless_split(seed, 3))
    else:
      rngs = list(jax.random.split(seed, 3))
  rng_shift, rng_zoom, rng_flip = rngs

  if method == _SHIFT:
    x = augment_shift(x, seed=rng_shift, **kwargs)
  elif method == _ZOOM_CROP:
    x = augment_zoom_crop(
        x, seed=rng_zoom, resize_method=resize_method, **kwargs)
  else:
    raise NotImplementedError(
        f"{method} is not supported for data augmentation.")
  if random_flip:
    x = tf.image.stateless_random_flip_left_right(x, rng_flip)
  return x


@gin.configurable()
def augment_shift(
    x: tf.Tensor,
    w: int = 4,
    seed: Union[None, jnp.ndarray, tf.Tensor] = None,
) -> tf.Tensor:
  """Randomly translates the image by w pixels.

  Args:
    x: Input image tensor.
    w: Padding size.
    seed: Random seed for augmenation.
  Returns:
    The augmented image tensor.
  """
  y = tf.pad(x, [[0] * 2, [w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
  return tf.image.stateless_random_crop(y, tf.shape(x), seed=seed)


def augment_zoom_crop(
    x: tf.Tensor,
    resize_method: str = "bilinear",
    zoom_ratio: float = 1.125,
    seed: Union[None, jnp.ndarray, tf.Tensor] = None,
) -> tf.Tensor:
  """Randomly zooms and crops the image.

  Args:
    x: Input image tensor.
    resize_method: Image resize method, 'bilinear' or 'nearest'.
    zoom_ratio: The ratio for zoom out. Default value is 9/8.
    seed: Random seed for augmenation.

  Returns:
    The augmented image tensor.

  Raises:
    NotImplementedError: An error occurred when resize method is not supported.
  """
  if resize_method != "nearest" and resize_method != "bilinear":
    raise NotImplementedError(f"{resize_method} is not supported.")
  imsize = float(tf.shape(x)[1])
  new_size = int(imsize * zoom_ratio)
  y = tf.image.resize(x, (new_size, new_size), method=resize_method)
  return tf.image.stateless_random_crop(y, tf.shape(x), seed=seed)
