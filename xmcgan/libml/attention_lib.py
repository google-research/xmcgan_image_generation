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

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from xmcgan.libml import losses

LARGE_NUM = 1e9


def cosine_similarity(x1, x2):
  """Calculates cosine similarity of two tensor."""
  dist = jnp.sum(jnp.multiply(x1, x2), -1)
  dist = dist / (jnp.linalg.norm(x1, axis=-1) * jnp.linalg.norm(x2, axis=-1))
  return dist


def l2_normalize(x, axis=None, epsilon=1e-12):
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return jnp.multiply(x, x_inv_norm)


def get_statistics(logits, labels):
  """Gets accuracy and entropy."""
  prob = jax.nn.softmax(logits)
  entropy = -jnp.mean(jnp.sum(prob * jnp.log(prob + 1e-8), axis=-1))
  label_acc = jnp.equal(
      jnp.argmax(logits, axis=-1), jnp.argmax(labels, axis=-1))
  label_acc = jnp.mean(jnp.asarray(label_acc, jnp.float32))
  return label_acc, entropy


def contrastive_loss(
    image_feat: jnp.ndarray,
    cond_feat: jnp.ndarray,
    l2_norm: bool = True,
    temperature: float = 0.1,
    sync_match: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Calculates contrastive loss."""

  if l2_norm:
    image_feat = l2_normalize(image_feat, -1)
    cond_feat = l2_normalize(cond_feat, -1)
  local_batch_size = image_feat.shape[0]
  if sync_match:
    raise NotImplementedError
  else:
    image_feat_large = image_feat
    cond_feat_large = cond_feat
    labels = jax.nn.one_hot(jnp.arange(local_batch_size), local_batch_size)
    logits_img2cond = jnp.matmul(image_feat,
                                 cond_feat_large.transpose()) / temperature
    logits_cond2img = jnp.matmul(cond_feat,
                                 image_feat_large.transpose()) / temperature
    loss_img2cond = losses.tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_img2cond)
    loss_cond2img = losses.tf_cross_entropy_loss_with_logits(
        labels=labels, logits=logits_cond2img)
    loss_img2cond = jnp.mean(loss_img2cond)
    loss_cond2img = jnp.mean(loss_cond2img)
    loss = loss_img2cond + loss_cond2img
    accuracy1, entropy1 = get_statistics(logits_img2cond, labels)
    accuracy2, entropy2 = get_statistics(logits_cond2img, labels)
    accuracy = 0.5 * (accuracy1 + accuracy2)
    entropy = 0.5 * (entropy1 + entropy2)
    return loss, accuracy, entropy


def attention_for_word(
    image_feat: jnp.ndarray,
    word_feat: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    l2_norm: bool = True,
    temperature: float = 0.1,
) -> jnp.ndarray:
  """Calculates attention for word"."""

  if l2_norm:
    image_feat = l2_normalize(image_feat, -1)
    word_feat = l2_normalize(word_feat, -1)
  # batch_size * region_num * word_num
  logits = jnp.matmul(image_feat, word_feat.transpose((0, 2, 1))) / temperature
  if mask is not None:
    logits = logits + mask * (-LARGE_NUM)
  # batch_size * region_num * word_num
  attn = jax.nn.softmax(logits, axis=-1)
  # batch_size * region_num * feat_dim
  region_context = jnp.matmul(attn, word_feat)
  return region_context


def attention(region_feat, word_feat, gamma, mask=None):
  """Calculates region attention.

  Args:
    region_feat: Regions features of shape (batch_size, region_num, feat_dim)
    word_feat: Word features of shape (batch_size, word_dim, feat_dim)
    gamma: Gamma used in the softmax shappenning function.
    mask: For masking padding word.

  Returns:
    region_context: For each word, its aggregated region context.
    alpha: The attention weights.
  """
  region_feat = l2_normalize(region_feat, -1)
  word_feat = l2_normalize(word_feat, -1)
  # batch_size * region_num * word_dim
  attn_matrix = jnp.matmul(region_feat, word_feat.transpose((0, 2, 1)))
  attn_matrix = attn_matrix * gamma
  if mask is not None:
    attn_matrix = attn_matrix + mask * (-1e9)
  alpha = jax.nn.softmax(attn_matrix, axis=-2)
  region_context = jnp.matmul(alpha.transpose((0, 2, 1)), region_feat)
  return region_context


def word_loss(image_feat, word_feat, max_len, gamma1=5, gamma2=5, gamma3=50):
  """Computes the word-level contrastive loss.


  Args:
    image_feat: Image features has shape (batch_size, region_num, feat_dim)
    word_feat: Word features has shape (batch_size, word_dim, feat_dim)
    max_len: The number of total words for each sentence. (batch_size,)
    gamma1: Gamma1 used in attnGAN paper.
    gamma2: Gamma2 used in attnGAN paper.
    gamma3: Gamma3 used in attnGAN paper.

  Returns:
    matching_loss: The word level matching loss.
    accuracy: The matching accuracy.
    entropy: The prediction entropy.
  """
  batch_size, region_num, _ = image_feat.shape
  total_len = word_feat.shape[1]

  def my_func(max_len_i, word_feat_i):
    word_feat_i = word_feat_i[None, :]
    word_feat_i = jnp.tile(word_feat_i, [batch_size, 1, 1])
    max_len_i = jnp.tile(max_len_i, region_num)
    mask = jnp.arange(
        total_len, dtype=jnp.float32)[None, :] >= max_len_i[:, None]
    mask = jnp.asarray(mask, dtype=jnp.float32)
    mask = mask[None, :]
    mask = jnp.tile(mask, (batch_size, 1, 1))
    mask_2 = mask[:, 0, :]
    # (batch_size, word_dim, feat_dim)
    region_context = attention(image_feat, word_feat_i, gamma1, mask)
    row_sim = cosine_similarity(word_feat_i, region_context)
    row_sim = row_sim * gamma2  # (batch_size, word_dim)
    row_sim = row_sim + mask_2 * (-1e9)
    row_sim = jax.scipy.special.logsumexp(row_sim, axis=-1, keepdims=True)
    row_sim = row_sim / gamma2
    return row_sim

  similarities = jax.vmap(my_func)(max_len, word_feat)
  similarities = similarities * gamma3
  similarities = jnp.squeeze(similarities)
  similarities_transpose = similarities  # To be consistent with tf
  similarities = similarities_transpose.transpose()

  labels = jax.nn.one_hot(jnp.arange(batch_size), batch_size)
  loss_0 = losses.tf_cross_entropy_loss_with_logits(
      labels=labels, logits=similarities)
  loss_1 = losses.tf_cross_entropy_loss_with_logits(
      labels=labels, logits=similarities_transpose)
  loss_0 = jnp.mean(loss_0)
  loss_1 = jnp.mean(loss_1)
  matching_loss = loss_0 + loss_1
  accuracy1, entropy1 = get_statistics(
      similarities, labels
  )  # different from tf, calculates accuracy and entropy from two sides
  accuracy2, entropy2 = get_statistics(
      similarities_transpose, labels
  )  # different from tf, calculates accuracy and entropy from two sides
  accuracy = 0.5 * (accuracy1 + accuracy2)
  entropy = 0.5 * (entropy1 + entropy2)
  return matching_loss, accuracy, entropy


def attention_for_g(region_feat, word_feat, gamma, mask=None):
  """Implements attention for each region.

  Args:
    region_feat: Region features of shape (batch_size, region_num, feat_dim)
    word_feat:  Word features of shape (batch_size, word_dim, feat_dim)
    gamma: Temperature  for the softmax.
    mask: Mask for the word

  Returns:
    region_context: Region context features of shape (batch_size, region_num,
    feat_dim).
    attn: Attention weights
  """
  # batch_size * region_num * word_dim
  region_feat = l2_normalize(region_feat, -1)
  word_feat = l2_normalize(word_feat, -1)
  attn_matrix = jnp.matmul(region_feat, word_feat.transpose((0, 2, 1)))
  attn_matrix = attn_matrix * gamma
  if mask is not None:
    attn_matrix = attn_matrix + mask * (-1e9)
  # batch_size * region_num * word_dim
  attn = jax.nn.softmax(attn_matrix)
  # batch_size * region_num * feat_dim
  region_context = jnp.matmul(attn, word_feat)
  return region_context, attn
