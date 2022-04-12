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
import logging
from typing import Tuple

from clu import deterministic_data
import jax
import ml_collections
import tensorflow as tf
from xmcgan.libml import coco_dataset

from tensorflow.python.data.experimental.ops import stats_aggregator

_CUSTOM_DATASETS = ["mscoco"]


def create_datasets(
    config: ml_collections.ConfigDict,
    data_rng) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations. See go/deterministic-training to
  learn how this helps with reproducible training.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.

  Returns:
    The training dataset, the evaluation dataset and num_training samples.
  """
  if config.batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({config.batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = config.batch_size // jax.device_count()
  per_device_batch_size_train = per_device_batch_size * config.d_step_per_g_step
  if config.dtype == "bfloat16":
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  if config.dataset == "mscoco":
    dataset_builder = coco_dataset.COCODataset(
        image_size=config.image_size,
        data_dtype=dtype,
        data_dir=config.data_dir,
        coco_version=config.coco_version,
        return_text=config.return_text,
        return_filename=config.return_filename)
    train_split = "train"
    eval_split = "val"
    preprocess_fn_train = functools.partial(
        dataset_builder.preprocess, training=True)
    preprocess_fn_eval = functools.partial(
        dataset_builder.preprocess, training=False)
    num_train_examples = dataset_builder.num_examples["train"]
  else:
    raise NotImplementedError

  train_data_rng, eval_data_rng = jax.random.split(data_rng, 2)

  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      rng=train_data_rng,
      preprocess_fn=preprocess_fn_train,
      cache=False,  # TODO(b/166121905): Re-enable caching.
      # We tell TFDS to not decode the images. This reduces the amount of data
      # in the cache (if activated above) and the shuffle buffer. We later
      # decode only the ramdom crop of the image we will be using.
      # decoders={"image": tfds.decode.SkipDecoding()},
      decoders=None,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=[jax.local_device_count(), per_device_batch_size_train],
      num_epochs=config.num_epochs if config.num_train_steps == -1 else None,
      shuffle=config.train_shuffle,
  )

  eval_num_batches = None
  eval_batch_size_per_replica = config.eval_batch_size // jax.device_count()
  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      rng=eval_data_rng,
      preprocess_fn=preprocess_fn_eval,
      # Only cache dataset in distributed setup to avoid consuming a lot of
      # memory in Colab and unit tests.
      cache=jax.host_count() > 1,
      batch_dims=[jax.local_device_count(), eval_batch_size_per_replica],
      num_epochs=None,
      shuffle_buffer_size=config.shuffle_buffer_size,
      shuffle=True,  # We need random order for dataset of imagenet
      pad_up_to_batches=eval_num_batches,
  )
  # Temporary workaround. See b/179292577.

  # Tmage1.0
  aggregator = tf.data.experimental.StatsAggregator()

  options = tf.data.Options()
  options.experimental_external_state_policy = (
      tf.data.experimental.ExternalStatePolicy.WARN)
  options.experimental_stats.aggregator = aggregator
  train_ds = train_ds.with_options(options)
  eval_ds = eval_ds.with_options(options)

  stats_summary = stats_aggregator.get_summary()
  logging.info(f'Stats summary {stats_summary}')

  # Tmage
  # logging.info(f'Train datset shape {train_ds.bufferSizeMin()}')
  # logging.info(f'Eval datset shape {eval_ds.bufferSizeMin()}')
  
  return train_ds, eval_ds, num_train_examples
