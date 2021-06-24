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

import abc
from typing import Optional

import tensorflow as tf
import tensorflow_datasets as tfds

# Size of TFRecord reader buffer, per file. (The TFRecordDataset default, 256KB,
# is too small and makes TPU trainers input-bound.)
_TFRECORD_READER_BUFFER_SIZE_BYTES = 64 * (1 << 20)  # 64 MiB


class BaseDataset(abc.ABC):
  """Base Dataset."""

  def __init__(self,
               image_size: int,
               num_classes: Optional[int] = None,
               z_dim: int = 128,
               z_generator: str = "cpu_generator"):
    """init function for BaseDataset.

    Args:
      image_size: Width and height of resized images.
      num_classes: Number of class labels in this dataset.
      z_dim: The dimension of random noise.
      z_generator: "cpu_generator" uses tf.random.Generator on cpu;
        "cpu_random" uses tf.random on cpu, otherwise use on device tf.random.
    """
    self.image_size = image_size
    self.num_classes = num_classes
    self.z_dim = z_dim
    self.z_generator = z_generator

  def as_dataset(self,
                 split: str,
                 shuffle_files: bool,
                 read_config: Optional[tfds.ReadConfig] = None,
                 decoders: Optional[tfds.decode.Decoder] = None):
    """Returns the split as a `tf.data.Dataset`."""
    del decoders  # Unused for now.
    if tf.io.gfile.glob(split):
      file_patterns = split
    else:
      file_patterns = self.get_file_patterns(split)
    assert tf.io.gfile.glob(file_patterns), (
        f"No data files matched {file_patterns}")
    shuffle_seed = None
    if read_config is not None:
      shuffle_seed = read_config.shuffle_seed
    files = tf.data.Dataset.list_files(
        file_patterns, shuffle=shuffle_files, seed=shuffle_seed)
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=16,
        num_parallel_calls=-1,
        deterministic=True)
    processed_ds = dataset.map(
        self._parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return processed_ds

  @abc.abstractmethod
  def _parse(self, example):
    """Returns a parsed, decoded `tf.data.Dataset`.

    Args:
      example: Scalar string tensor representing bytes data.

    Returns:
      outputs: Dict of string feature names to Tensor feature values.
    """
    pass

  def preprocess(self, features, training=True):
    """Per-example preprocessing operations."""
    del training
    return features

  @abc.abstractmethod
  def get_file_patterns(self,
                        split: Optional[str] = None,
                        file_pattern: Optional[str] = None):
    pass

  @property
  @abc.abstractmethod
  def num_examples(self):
    pass
