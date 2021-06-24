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

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class ConvBatchNormReluBlock(nn.Module):
  """Basic building block of Inception V3."""
  filters: int
  kernel_size: Tuple[int, int]
  strides: Tuple[int, int]
  padding: str
  use_running_average: bool = True

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=self.filters,
        kernel_size=self.kernel_size,
        strides=self.strides,
        padding=self.padding,
        use_bias=False)(
            x)
    x = nn.BatchNorm(
        use_running_average=self.use_running_average,
        epsilon=1e-3,
        use_scale=False,
        use_bias=True)(
            x)
    x = nn.relu(x)
    return x


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.
  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
      (1,) + strides + (1,), padding)
  return pool_sum / pool_denom


class InceptionV3(nn.Module):
  """The Inception V3 for Flax.

  use_running_average is passed to each Batch Normalization layer;
  if True it does not update the stats parameters of the Batch Normalization
  layers.
  """
  use_running_average: bool = True
  include_top: bool = False

  @nn.compact
  def __call__(self, x):
    x = ConvBatchNormReluBlock(32, (3, 3), (2, 2), 'VALID',
                               self.use_running_average)(
                                   x)
    x = ConvBatchNormReluBlock(32, (3, 3), (1, 1), 'VALID',
                               self.use_running_average)(
                                   x)
    x = ConvBatchNormReluBlock(64, (3, 3), (1, 1), 'SAME',
                               self.use_running_average)(
                                   x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='VALID')
    x = ConvBatchNormReluBlock(80, (1, 1), (1, 1), 'VALID',
                               self.use_running_average)(
                                   x)
    x = ConvBatchNormReluBlock(192, (3, 3), (1, 1), 'VALID',
                               self.use_running_average)(
                                   x)
    x = nn.max_pool(x, (3, 3), (2, 2), 'VALID')

    branch1x1 = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)

    branch5x5 = ConvBatchNormReluBlock(48, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch5x5 = ConvBatchNormReluBlock(64, (5, 5), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch5x5)

    branch3x3 = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch3x3 = ConvBatchNormReluBlock(96, (3, 3), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch3x3)
    branch3x3 = ConvBatchNormReluBlock(96, (3, 3), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch3x3)

    branch_pool = tensorflow_style_avg_pooling(
        x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
    branch_pool = ConvBatchNormReluBlock(32, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch_pool)

    # mixed0 in the Keras Model
    x = jnp.concatenate((branch1x1, branch5x5, branch3x3, branch_pool), axis=-1)

    branch1x1 = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)

    branch5x5 = ConvBatchNormReluBlock(48, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch5x5 = ConvBatchNormReluBlock(64, (5, 5), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch5x5)

    branch3x3 = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch3x3 = ConvBatchNormReluBlock(96, (3, 3), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch3x3)
    branch3x3 = ConvBatchNormReluBlock(96, (3, 3), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch3x3)

    branch_pool = tensorflow_style_avg_pooling(
        x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
    branch_pool = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch_pool)

    # mixed 1 in Keras Model
    x = jnp.concatenate((branch1x1, branch5x5, branch3x3, branch_pool), axis=-1)

    branch1x1 = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)

    branch5x5 = ConvBatchNormReluBlock(48, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch5x5 = ConvBatchNormReluBlock(64, (5, 5), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch5x5)

    branch3x3 = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch3x3 = ConvBatchNormReluBlock(96, (3, 3), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch3x3)
    branch3x3 = ConvBatchNormReluBlock(96, (3, 3), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch3x3)

    branch_pool = tensorflow_style_avg_pooling(
        x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
    branch_pool = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch_pool)

    # mixed 2 in Keras Model
    x = jnp.concatenate((branch1x1, branch5x5, branch3x3, branch_pool), axis=-1)

    branch3x3 = ConvBatchNormReluBlock(384, (3, 3), (2, 2), 'VALID',
                                       self.use_running_average)(
                                           x)

    branch3x3dbl = ConvBatchNormReluBlock(64, (1, 1), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              x)
    branch3x3dbl = ConvBatchNormReluBlock(96, (3, 3), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch3x3dbl)
    branch3x3dbl = ConvBatchNormReluBlock(96, (3, 3), (2, 2), 'VALID',
                                          self.use_running_average)(
                                              branch3x3dbl)

    branch_pool = nn.max_pool(x, (3, 3), (2, 2), 'VALID')

    # mixed3 in Keras Model
    x = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=-1)

    branch1x1 = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)

    branch7x7 = ConvBatchNormReluBlock(128, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch7x7 = ConvBatchNormReluBlock(128, (1, 7), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch7x7)
    branch7x7 = ConvBatchNormReluBlock(192, (7, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch7x7)

    branch7x7dbl = ConvBatchNormReluBlock(128, (1, 1), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              x)
    branch7x7dbl = ConvBatchNormReluBlock(128, (7, 1), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)
    branch7x7dbl = ConvBatchNormReluBlock(128, (1, 7), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)
    branch7x7dbl = ConvBatchNormReluBlock(128, (7, 1), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)
    branch7x7dbl = ConvBatchNormReluBlock(192, (1, 7), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)

    branch_pool = tensorflow_style_avg_pooling(
        x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
    branch_pool = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch_pool)

    # mixed4 in Keras Model
    x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool),
                        axis=-1)

    # mixed5 & mixed6 are built via this loop
    for _ in range(2):
      branch1x1 = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             x)

      branch7x7 = ConvBatchNormReluBlock(160, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             x)
      branch7x7 = ConvBatchNormReluBlock(160, (1, 7), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch7x7)
      branch7x7 = ConvBatchNormReluBlock(192, (7, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch7x7)

      branch7x7dbl = ConvBatchNormReluBlock(160, (1, 1), (1, 1), 'SAME',
                                            self.use_running_average)(
                                                x)
      branch7x7dbl = ConvBatchNormReluBlock(160, (7, 1), (1, 1), 'SAME',
                                            self.use_running_average)(
                                                branch7x7dbl)
      branch7x7dbl = ConvBatchNormReluBlock(160, (1, 7), (1, 1), 'SAME',
                                            self.use_running_average)(
                                                branch7x7dbl)
      branch7x7dbl = ConvBatchNormReluBlock(160, (7, 1), (1, 1), 'SAME',
                                            self.use_running_average)(
                                                branch7x7dbl)
      branch7x7dbl = ConvBatchNormReluBlock(192, (1, 7), (1, 1), 'SAME',
                                            self.use_running_average)(
                                                branch7x7dbl)

      branch_pool = tensorflow_style_avg_pooling(
          x, window_shape=(3, 3), strides=(1, 1), padding='SAME')

      branch_pool = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                           self.use_running_average)(
                                               branch_pool)
      x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool),
                          axis=-1)

    branch1x1 = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)

    branch7x7 = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch7x7 = ConvBatchNormReluBlock(192, (1, 7), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch7x7)
    branch7x7 = ConvBatchNormReluBlock(192, (7, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           branch7x7)

    branch7x7dbl = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              x)
    branch7x7dbl = ConvBatchNormReluBlock(192, (7, 1), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)
    branch7x7dbl = ConvBatchNormReluBlock(192, (1, 7), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)
    branch7x7dbl = ConvBatchNormReluBlock(192, (7, 1), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)
    branch7x7dbl = ConvBatchNormReluBlock(192, (1, 7), (1, 1), 'SAME',
                                          self.use_running_average)(
                                              branch7x7dbl)

    branch_pool = tensorflow_style_avg_pooling(
        x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
    branch_pool = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch_pool)

    # mixed7 in Keras Model
    x = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool),
                        axis=-1)

    branch3x3 = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                       self.use_running_average)(
                                           x)
    branch3x3 = ConvBatchNormReluBlock(320, (3, 3), (2, 2), 'VALID',
                                       self.use_running_average)(
                                           branch3x3)

    branch7x7x3 = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             x)
    branch7x7x3 = ConvBatchNormReluBlock(192, (1, 7), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch7x7x3)
    branch7x7x3 = ConvBatchNormReluBlock(192, (7, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             branch7x7x3)
    branch7x7x3 = ConvBatchNormReluBlock(192, (3, 3), (2, 2), 'VALID',
                                         self.use_running_average)(
                                             branch7x7x3)

    branch_pool = nn.max_pool(x, (3, 3), (2, 2), 'VALID')

    # mixed8 in Keras Model
    x = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=-1)

    # produces the layers mixed9, mixed10
    for _ in range(2):
      branch1x1 = ConvBatchNormReluBlock(320, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             x)

      branch3x3 = ConvBatchNormReluBlock(384, (1, 1), (1, 1), 'SAME',
                                         self.use_running_average)(
                                             x)
      branch3x3_1 = ConvBatchNormReluBlock(384, (1, 3), (1, 1), 'SAME',
                                           self.use_running_average)(
                                               branch3x3)
      branch3x3_2 = ConvBatchNormReluBlock(384, (3, 1), (1, 1), 'SAME',
                                           self.use_running_average)(
                                               branch3x3)
      branch3x3 = jnp.concatenate((branch3x3_1, branch3x3_2), axis=-1)

      branch3x3dbl = ConvBatchNormReluBlock(448, (1, 1), (1, 1), 'SAME',
                                            self.use_running_average)(
                                                x)
      branch3x3dbl = ConvBatchNormReluBlock(384, (3, 3), (1, 1), 'SAME',
                                            self.use_running_average)(
                                                branch3x3dbl)
      branch3x3dbl_1 = ConvBatchNormReluBlock(384, (1, 3), (1, 1), 'SAME',
                                              self.use_running_average)(
                                                  branch3x3dbl)
      branch3x3dbl_2 = ConvBatchNormReluBlock(384, (3, 1), (1, 1), 'SAME',
                                              self.use_running_average)(
                                                  branch3x3dbl)
      branch3x3dbl = jnp.concatenate((branch3x3dbl_1, branch3x3dbl_2), axis=-1)

      branch_pool = tensorflow_style_avg_pooling(
          x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
      branch_pool = ConvBatchNormReluBlock(192, (1, 1), (1, 1), 'SAME',
                                           self.use_running_average)(
                                               branch_pool)

      # layer mixed+str(9+i) in the Keras Model
      x = jnp.concatenate((branch1x1, branch3x3, branch3x3dbl, branch_pool),
                          axis=-1)

    # if the top dense layer is included then a spatial average
    # pooling is done.
    pool = None
    if self.include_top:
      # do mean on the spatial dimensions
      x = jnp.mean(x, axis=(1, 2))
      pool = x
      x = nn.Dense(1000)(x)

    return pool, x
