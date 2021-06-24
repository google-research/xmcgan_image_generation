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

import functools
from typing import Any
import flax.linen as nn
from jax.nn.initializers import glorot_normal
import jax.numpy as jnp
import ml_collections

from xmcgan.libml import attention_lib as attn_lib
from xmcgan.libml import layers
from xmcgan.nets import common
from xmcgan.utils import device_utils


class Discriminator(nn.Module):
  """Basic Discriminator Structure."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu

  def setup(self):
    self.image_size = self.config.image_size
    self.df_dim = self.config.df_dim
    self.spectral_norm = self.config.d_spectral_norm
    self.word_contrastive = self.config.word_contrastive
    self.sentence_contrastive = self.config.sentence_contrastive
    self.image_contrastive = self.config.image_contrastive
    self.cond_size = self.config.cond_size

  @nn.compact
  def __call__(self, inputs):
    """Runs a forward pass of the XMC-GAN discriminator.

    Args:
      inputs: Tuple consisting of the concatenation of the real / generated
        images and an input batch as a dictionary.
    Returns:
      out: Final output logits of the discriminator.
      statistic_dict: Dictionary consisting of contrastive losses and training
        statistics.
    """
    x, cond_dict = inputs
    cond = cond_dict["sentence_embedding"]
    word_feat = cond_dict["embedding"]
    max_len = cond_dict["max_len"]
    fake_word_loss, fake_word_acc, fake_word_entropy = 0, 0, 0
    real_word_loss, real_word_acc, real_word_entropy = 0, 0, 0
    real_sentence_loss, real_sentence_acc, real_sentence_entropy = 0, 0, 0
    fake_sentence_loss, fake_sentence_acc, fake_sentence_entropy = 0, 0, 0
    image_contrastive_loss, image_contrastive_acc, image_contrastive_entropy = 0, 0, 0
    if self.spectral_norm:
      conv_fn = functools.partial(
          layers.SpectralConv,
          train=self.train,
          dtype=self.dtype,
          kernel_init=glorot_normal())
      dense_fn = functools.partial(
          layers.SpectralDense,
          train=self.train,
          dtype=self.dtype,
          kernel_init=glorot_normal())
    else:
      conv_fn = functools.partial(
          nn.Conv, dtype=self.dtype, kernel_init=glorot_normal())
      dense_fn = functools.partial(
          nn.Dense, dtype=self.dtype, kernel_init=glorot_normal())
    if self.image_size == 128:
      channel_dims = [2, 4, 8, 16, 16]
      downsamples = [True, True, True, True, False]
    elif self.image_size == 256:
      channel_dims = [2, 4, 8, 8, 16, 16]
      downsamples = [True, True, True, True, True, False]
    block_args = dict(
        activation_fn=self.activation_fn, conv_fn=conv_fn, dtype=self.dtype)
    x = common.DiscOptimizedBlock(self.df_dim, **block_args)(x)
    for c_ratio, downsample in zip(channel_dims, downsamples):
      x = common.DiscBlock(
          self.df_dim * c_ratio, downsample=downsample, **block_args)(
              x)
      if x.shape[1] == self.cond_size:
        x_cond = x

    x = self.activation_fn(x)
    x_pool = jnp.sum(x, axis=(1, 2))
    out = dense_fn(1)(x_pool)
    embedding = dense_fn(self.df_dim * channel_dims[-1], use_bias=True)(cond)
    sent_cond = embedding
    tile_num = x_pool.shape[0] // embedding.shape[0]
    embedding = jnp.tile(embedding, (tile_num, 1))
    out += jnp.sum(x_pool * embedding, axis=1, keepdims=True)
    if self.sentence_contrastive:
      real_feat, fake_feat = jnp.split(
          x_pool, 2)  # Different from tf,as real image is the first half.
      fake_sentence_loss, fake_sentence_acc, fake_sentence_entropy = attn_lib.contrastive_loss(
          fake_feat, sent_cond)
      real_sentence_loss, real_sentence_acc, real_sentence_entropy = attn_lib.contrastive_loss(
          real_feat, sent_cond)
    if self.word_contrastive:
      embedding_dim = word_feat.shape[-1]
      x_cond = conv_fn(embedding_dim, kernel_size=(1, 1))(x_cond)
      total_region_size = self.cond_size * self.cond_size
      x_cond_reshape = x_cond.reshape([-1, total_region_size, embedding_dim])
      real_x_cond, fake_x_cond = jnp.split(x_cond_reshape, 2)
      fake_word_loss, fake_word_acc, fake_word_entropy = attn_lib.word_loss(
          fake_x_cond, word_feat, max_len)
      real_word_loss, real_word_acc, real_word_entropy = attn_lib.word_loss(
          real_x_cond, word_feat, max_len)
    if self.image_contrastive:
      real_feat, fake_feat = jnp.split(x_pool, 2)
      image_contrastive_loss, image_contrastive_acc, image_contrastive_entropy = \
          attn_lib.contrastive_loss(fake_feat, real_feat)
    statistic_dict = dict(
        fake_word_loss=fake_word_loss,
        fake_word_acc=fake_word_acc,
        fake_word_entropy=fake_word_entropy,
        real_word_loss=real_word_loss,
        real_word_acc=real_word_acc,
        real_word_entropy=real_word_entropy,
        fake_sentence_loss=fake_sentence_loss,
        fake_sentence_acc=fake_sentence_acc,
        fake_sentence_entropy=fake_sentence_entropy,
        real_sentence_loss=real_sentence_loss,
        real_sentence_acc=real_sentence_acc,
        real_sentence_entropy=real_sentence_entropy,
        image_contrastive_loss=image_contrastive_loss,
        image_contrastive_acc=image_contrastive_acc,
        image_contrastive_entropy=image_contrastive_entropy)
    return out, statistic_dict


class Generator(nn.Module):
  """Generator Network."""
  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu

  def setup(self):
    self.image_size = self.config.image_size
    self.gf_dim = self.config.gf_dim
    self.spectral_norm = self.config.g_spectral_norm
    self.batch_norm_group_size = self.config.batch_norm_group_size
    self.gamma = self.config.gamma_for_g

  @nn.compact
  def __call__(self, inputs):
    """Runs a forward pass of the XMC-GAN generator.

    Args:
      inputs: Tuple consisting of the input batch as a dictionary and a noise
        vector.
    Returns:
      generated_image: Tensor of generated image with values in [0, 1].
    """
    cond_dict, z = inputs
    cond = cond_dict["sentence_embedding"]
    word_feat = cond_dict["embedding"]
    max_len = cond_dict["max_len"]
    embedding_dim = word_feat.shape[-1]
    batch_size = z.shape[0]

    if self.spectral_norm:
      conv_fn = functools.partial(
          layers.SpectralConv,
          train=self.train,
          dtype=self.dtype,
          kernel_init=glorot_normal())
      dense_fn = functools.partial(
          layers.SpectralDense,
          train=self.train,
          dtype=self.dtype,
          kernel_init=glorot_normal())
    else:
      conv_fn = functools.partial(
          nn.Conv, dtype=self.dtype, kernel_init=glorot_normal())
      dense_fn = functools.partial(
          nn.Dense, dtype=self.dtype, kernel_init=glorot_normal())
    norm_fn = functools.partial(
        nn.BatchNorm,
        use_running_average=not self.train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name="batch" if self.batch_norm_group_size > 0 else None,
        axis_index_groups=device_utils.get_device_groups(
            self.batch_norm_group_size, z.shape[0])
        if self.train and self.batch_norm_group_size > 0 else None,
        dtype=self.dtype)
    if self.image_size == 256:
      channel_dims = [16, 8, 8, 4, 2, 1]
    elif self.image_size == 128:
      channel_dims = [16, 8, 4, 2, 1]
    block_args = dict(
        dense_fn=dense_fn,
        conv_fn=conv_fn,
        activation_fn=self.activation_fn,
        norm_fn=norm_fn,
        dtype=self.dtype)
    z_dim = z.shape[-1]
    global_cond = dense_fn(z_dim)(cond)
    global_cond = jnp.concatenate([global_cond, z], axis=-1)
    x = dense_fn(self.gf_dim * 16 * 4 * 4)(z)
    x = jnp.reshape(x, (-1, 4, 4, self.gf_dim * 16))
    for i in range(2):
      x = common.GenBlock(self.gf_dim * channel_dims[i],
                          **block_args)(x, global_cond)  # 16x16
    x_cond = conv_fn(embedding_dim, kernel_size=(1, 1))(x)
    spatial_size = x_cond.shape[1]
    total_region_size = spatial_size * spatial_size
    total_len = word_feat.shape[1]
    x_cond = jnp.reshape(x_cond, [batch_size, total_region_size, embedding_dim])
    mask = jnp.arange(total_len, dtype=jnp.float32)[None, :] >= max_len
    mask = jnp.asarray(mask, jnp.float32)
    mask = jnp.expand_dims(mask, 1)
    mask = jnp.tile(mask, [1, total_region_size, 1])
    region_context, _ = attn_lib.attention_for_g(x_cond, word_feat, self.gamma,
                                                 mask)
    region_context = jnp.reshape(
        region_context, [batch_size, spatial_size, spatial_size, embedding_dim])
    spatial_cond = jnp.reshape(global_cond, [batch_size, 1, 1, -1])
    spatial_cond = jnp.tile(spatial_cond, [1, spatial_size, spatial_size, 1])
    spatial_cond = jnp.concatenate([region_context, spatial_cond], axis=-1)
    for i in range(2, len(channel_dims)):
      spatial_cond_upsample = common.upsample(spatial_cond)
      x = common.GenSpatialBlock(self.gf_dim * channel_dims[i],
                                 **block_args)(x, spatial_cond,
                                               spatial_cond_upsample)
      spatial_cond = spatial_cond_upsample
    x = layers.LocalConditionalBatchNorm(
        norm_fn=norm_fn, conv_fn=conv_fn)(x, spatial_cond)
    x = self.activation_fn(x)
    x = conv_fn(3, kernel_size=(3, 3))(x)
    x = jnp.tanh(x)
    x = (x + 1.0) / 2.0
    return x
