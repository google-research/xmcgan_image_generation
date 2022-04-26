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
from typing import Any, Dict, Optional, Tuple, Union

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
import imageio
from jax.experimental import host_callback as hcb

from xmcgan.utils import inception_utils


class EvalMetric:
  """Evaluation class for FID and Inception Score.

  Attributes:
    ds: A tf.data.Dataset object for providing the evaluation data.
    config: An ml_collections.ConfigDict object containing model configuration.
    eval_num: A integer count for the number of images for calculating the
      inception score and FID, preferbaly bigger than 10000.
    eval_batch_size: Batch size for each forward operation.
    strategy: A tf.distribute.Strategy object for TPU/GPU processing.
    avg_num: A integer count for the number of calucation to get the mean and
      standard deviation for FID and Inception Score.
    num_splits: A integer count for the number of splits for eval_num for the
      calculation.
    inception_ckpt_path: Optionally provide the path to the pretrained Inception
      model to be used for computing IS / FID.
  """

  def __init__(self,
               ds: tf.data.Dataset,
               config: ml_collections.ConfigDict,
               num_splits: int = 1,
               inception_ckpt_path: Optional[str] = None) -> None:
    """Creates a new metric evaluation object."""

    self.ds = ds
    self.config = config
    self.eval_num = config.eval_num
    self.eval_batch_size = config.eval_batch_size
    self.avg_num = config.eval_avg_num
    self.num_splits = num_splits
    (self._inception_model, self._inception_params
     ) = inception_utils.inception_model(inception_ckpt_path)
    inception_func = functools.partial(
        inception_utils.get_inception,
        model=self._inception_model,
        model_params=self._inception_params)
    self._p_get_inception = jax.pmap(
        lambda x: jax.lax.all_gather(inception_func(x), axis_name="batch"),
        axis_name="batch")
    # Calculates pooling feature for real image only once to save time.
    self._pool = self._get_real_pool_for_evaluation()

  def _get_real_pool_for_evaluation(self):
    """Gets numpy arrays for pooling features and logits for real images."""
    logging.info("Get pool for %d samples", self.eval_num)

    n_iter = (self.eval_num // self.eval_batch_size) + 1
    pool = []
    for _ in range(n_iter):
      inputs = jax.tree_map(np.asarray, next(self.ds))  # pytype: disable=wrong-arg-types
      image = inputs["image"]
      pool_val, _ = self._p_get_inception(image)
      pool_val = pool_val[0]
      pool.append(pool_val)
    pool_total = jnp.concatenate(pool, 1)
    pool_total = jnp.reshape(pool_total, (-1, 2048))
    pool_total = pool_total[0:self.eval_num]
    logging.info("Active Evaluation Size For Real Data: %d",
                 pool_total.shape[0])
    return pool_total

  def _get_generated_images(
      self, rng: np.ndarray, state: Any, batch: Dict[str, jnp.ndarray],
      generator: Union[nn.Module, functools.partial],
      config: ml_collections.ConfigDict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Gets pooling/logtis features for generated images per batch.

    Args:
      rng: Random number generator function.
      state: Model checkpoint state.
      batch: Batch of examples.
      generator: Generator module that outputs generated images.
      config: Model configuration file.

    Returns:
      generated_image: [batch_size, H, W, 3] array with values in [0, 1].
      ema_generated_image: [batch_size, H, W, 3] array with values in [0, 1].
    """
    def jax_save(file, arr):
      def save_to_file(a, transforms):
          print("save image")
          jax.numpy.save('/images/'+file[i], a[i])
      hcb.id_tap(save_to_file, arr)

    if config.dtype == "bfloat16":
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32
    z = jax.random.normal(
        rng, (batch["image"].shape[0], config.z_dim), dtype=dtype)
    g_variables = {"params": state.g_optimizer.target}
    ema_g_variables = {"params": state.ema_params}
    g_variables.update(state.generator_state)
    ema_g_variables.update(state.generator_state)

    generated_image = generator().apply(g_variables, (batch, z), mutable=False)
    ema_generated_image = generator().apply(
        ema_g_variables, (batch, z), mutable=False)
    generated_image = jnp.asarray(generated_image, jnp.float32)
    ema_generated_image = jnp.asarray(ema_generated_image, jnp.float32)

    filenames = batch["filename"]
    print("Save batches")

    B, _, _, _ = generated_image.shape()
    for i in range(B):
      jax_save(filenames[i], generated_image[i])

    return generated_image, ema_generated_image

  def _get_generated_pool_for_evaluation(self,
                                         generator_fn: Union[nn.Module,
                                                             functools.partial],
                                         state: Any, rng: np.ndarray):
    """Gets numpy arrays for pooling features and logits for generated images."""
    n_iter = (self.eval_num // self.eval_batch_size) + 1
    pool, logits = [], []
    ema_pool, ema_logits = [], []
    p_generate_batch = jax.pmap(
        functools.partial(
            self._get_generated_images,
            generator=generator_fn,
            config=self.config,
        ),
        axis_name="batch")
    for step in range(n_iter):
      inputs = jax.tree_map(np.asarray, next(self.ds))  # pytype: disable=wrong-arg-types
    
      step_sample_batch_rng = jax.random.fold_in(rng, step)
      step_sample_batch_rngs = jax.random.split(step_sample_batch_rng,
                                                jax.local_device_count())
      generated_image, ema_generated_image = p_generate_batch(
          step_sample_batch_rngs, state,
          jax.tree_map(np.asarray, inputs))  # (1, 4, 128, 128, 3)
      pool_val, pool_logits = self._p_get_inception(generated_image)
      pool.append(pool_val[0])
      logits.append(pool_logits[0])
      ema_pool_val, ema_pool_logits = self._p_get_inception(ema_generated_image)
      ema_pool.append(ema_pool_val[0])
      ema_logits.append(ema_pool_logits[0])

    pool_total = jnp.concatenate(pool, 1)
    pool_total = jnp.reshape(pool_total, (-1, 2048))
    pool_total = pool_total[0:self.eval_num]
    logits_total = jnp.concatenate(logits, 1)
    logits_total = jnp.reshape(logits_total, (-1, 1000))
    logits_total = logits_total[0:self.eval_num]
    ema_pool_total = jnp.concatenate(ema_pool, 1)
    ema_pool_total = jnp.reshape(ema_pool_total, (-1, 2048))
    ema_pool_total = ema_pool_total[0:self.eval_num]
    ema_logits_total = jnp.concatenate(ema_logits, 1)
    ema_logits_total = jnp.reshape(ema_logits_total, (-1, 1000))
    ema_logits_total = ema_logits_total[0:self.eval_num]
    logging.info("Active Evaluation Size For Fake Data: %d, %d, %d, %d",
                 pool_total.shape[0], ema_pool_total.shape[0],
                 logits_total.shape[0], ema_logits_total.shape[0])
    return pool_total, logits_total, ema_pool_total, ema_logits_total

  def calculate_inception_fid(self, generator_fn: Union[nn.Module,
                                                        functools.partial],
                              state: Any, rng: np.ndarray):
    """Calculates Inception score and FID.

    Args:
      generator_fn: A generator for producing synthetic images.
      state: Training state data structure for storing model checkpoints.
      rng: Random number generator.

    Returns:
      fid: The average FID score for the generated images.
      fid_std: The standard deviation of FID for the generated images.
      inception_score: The average Inception Score for the generated images.
      inception_score_std: The standard deviation of Inception Score for the
        generated images.
    """
    fid_list, inception_list = [], []
    ema_fid_list, ema_inception_list = [], []
    logging.info("Calculate Generator Statistics")
    for i in range(self.avg_num):
      avg_rng = jax.random.fold_in(rng, i)
      generated_pool, generated_logits, ema_generated_pool, ema_generated_logits = \
          self._get_generated_pool_for_evaluation(generator_fn, state, avg_rng)
      inception_score, _ = inception_utils.calculate_inception_score(
          generated_logits, num_splits=self.num_splits)
      ema_inception_score, _ = inception_utils.calculate_inception_score(
          ema_generated_logits, num_splits=self.num_splits)
      fid = inception_utils.calculate_fid(generated_pool, self._pool)
      ema_fid = inception_utils.calculate_fid(ema_generated_pool, self._pool)
      fid_list.append(fid)
      inception_list.append(inception_score)
      ema_fid_list.append(ema_fid)
      ema_inception_list.append(ema_inception_score)

    fid, fid_std = np.mean(fid_list), np.std(fid_list)
    inception_score, inception_score_std = \
            np.mean(inception_list), np.std(inception_list)
    ema_fid, ema_fid_std = np.mean(ema_fid_list), np.std(ema_fid_list)
    ema_inception_score, ema_inception_score_std = \
            np.mean(ema_inception_list), np.std(ema_inception_list)

    return (fid, fid_std, inception_score, inception_score_std, ema_fid,
            ema_fid_std, ema_inception_score, ema_inception_score_std)

