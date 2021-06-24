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

from typing import Any
import flax.linen as nn
import jax
import jax.numpy as jnp

from xmcgan.libml import layers


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

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


def upsample(x, factor=2):
  n, h, w, c = x.shape
  x = jax.image.resize(x, (n, h * factor, w * factor, c), method="nearest")
  return x


def dsample(x):
  return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding="same")


class DiscBlock(nn.Module):
  """Discriminator Basic Block."""
  filters: int
  downsample: bool
  conv_fn: Any
  activation_fn: Any = nn.relu
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x):
    needs_projection = self.downsample or x.shape[-1] != self.filters
    x0 = x
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    if needs_projection:
      x0 = self.conv_fn(self.filters, kernel_size=(1, 1))(x0)
    if self.downsample:
      x = dsample(x)
      x0 = dsample(x0)
    return x0 + x


class DiscBlockDeep(nn.Module):
  """Discriminator Deep Block."""
  filters: int
  downsample: bool
  conv_fn: Any
  bottle_neck_ratio: int = 4
  activation_fn: Any = nn.relu
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x):
    in_channels = x.shape[-1]
    hidden_channels = self.filters // self.bottle_neck_ratio
    learnable_sc = True if (in_channels != self.filters) else False
    residual = x

    x = self.activation_fn(x)
    x = self.conv_fn(hidden_channels, kernel_size=(1, 1), name="conv0")(x)
    x = self.activation_fn(x)
    x = self.conv_fn(hidden_channels, kernel_size=(3, 3), name="conv1")(x)
    x = self.activation_fn(x)
    x = self.conv_fn(hidden_channels, kernel_size=(3, 3), name="conv2")(x)
    x = self.activation_fn(x)
    if self.downsample:
      residual = dsample(residual)
      x = dsample(x)
    x = self.conv_fn(self.filters, kernel_size=(1, 1), name="conv3")(x)
    if learnable_sc:
      residual_concat = self.conv_fn(
          self.filters - in_channels, kernel_size=(1, 1), name="conv_sc")(
              residual)
      residual = jnp.concatenate([residual, residual_concat], axis=-1)
    return x + residual


class DiscOptimizedBlock(nn.Module):
  """Discriminator Optimized Block."""
  filters: int
  conv_fn: Any
  activation_fn: Any = nn.relu
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x):
    x0 = x
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    x = dsample(x)
    x0 = dsample(x0)
    x0 = self.conv_fn(self.filters, kernel_size=(1, 1))(x0)
    return x + x0


class GenBlock(nn.Module):
  """Generator Basic blocks."""
  filters: int
  conv_fn: Any
  dense_fn: Any
  norm_fn: Any
  activation_fn: Any = nn.relu
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x, cond):
    x0 = x
    x = layers.ConditionalBatchNorm(
        norm_fn=self.norm_fn, dense_fn=self.dense_fn)(x, cond)
    x = self.activation_fn(x)
    x = upsample(x)
    # use bias=True to be same as tf model
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=True)(x)
    x = layers.ConditionalBatchNorm(
        norm_fn=self.norm_fn, dense_fn=self.dense_fn)(x, cond)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=True)(x)
    x0 = upsample(x0)
    x0 = self.conv_fn(self.filters, kernel_size=(1, 1), use_bias=True)(x0)
    return x + x0


class GenSpatialBlock(nn.Module):
  """Generator Deep Spatial blocks."""
  filters: int
  conv_fn: Any
  dense_fn: Any
  norm_fn: Any
  activation_fn: Any = nn.relu
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x, cond0, cond1):
    x0 = x
    x = layers.LocalConditionalBatchNorm(
        norm_fn=self.norm_fn, conv_fn=self.conv_fn)(x, cond0)
    x = self.activation_fn(x)
    x = upsample(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=True)(x)
    x = layers.LocalConditionalBatchNorm(
        norm_fn=self.norm_fn, conv_fn=self.conv_fn)(x, cond1)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=True)(x)
    x0 = upsample(x0)
    x0 = self.conv_fn(self.filters, kernel_size=(1, 1), use_bias=True)(x0)
    return x + x0
