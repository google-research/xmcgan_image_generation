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

from typing import Dict, Optional
import tensorflow as tf

from xmcgan.libml import augmentation
from xmcgan.libml import base_dataset
from xmcgan.libml import dataset_constants

_DEFAULT_STORAGE_DIR = "xmcgan/data/"
_VALID_DATASET_VERSIONS = ("2014","2014-tmage")


class COCODataset(base_dataset.BaseDataset):
  """Dataset class for preprocessing data in the JAX expected format."""

  def __init__(
      self,
      image_size: int = 128,
      data_dtype: tf.dtypes.DType = tf.float32,
      num_classes: int = 90,
      data_dir: str = _DEFAULT_STORAGE_DIR,
      coco_version: str = "2017",
      sentence_num: int = 5,
      select_num: int = 1,
      bert_dim: int = dataset_constants.PRETRAINED_BERT_DIM,
      return_text: bool = False,
      return_filename: bool = False,
      **kwargs,
  ):
    super().__init__(image_size=image_size, num_classes=num_classes, **kwargs)
    self.data_dir = data_dir
    if coco_version not in _VALID_DATASET_VERSIONS:
      raise ValueError(
          f"coco_version must be one of {_VALID_DATASET_VERSIONS}.")
    self.coco_version = coco_version
    if data_dtype not in [tf.float32, tf.bfloat16]:
      raise ValueError("data_dtype must be either tf.float32 or tf.bfloat16.")

    self.data_dtype = data_dtype
    self.sentence_num = sentence_num
    self.return_text = return_text
    self.return_filename = return_filename
    self.max_text_length = dataset_constants.COCO_MAX_TEXT_LENGTH

    # Set dataset specific settings.
    if coco_version == "ln":
      # Most LN images only have 1 caption.
      self.max_text_length = dataset_constants.LN_MAX_TEXT_LENGTH
      self.sentence_num = 1

    self.embeddding_shape = [self.sentence_num, self.max_text_length, bert_dim]
    self.caption_id_shape = [self.sentence_num, self.max_text_length]
    self.select_num = select_num

  def _parse(self, example):
    """Returns a parsed, preprocessed, and batched `tf.data.Dataset`.

    Args:
      example: Scalar string tensor representing bytes data.

    Returns:
      outputs: Dict of string feature names to Tensor feature values:
        "caption/max_len": tf.int64 tensor of shape (self.sentence_num,)
          indicating the actual caption length for a given image..
        "image": tf.float32 tensor of shape (H, W, C), normalized to
          have values in [0, 1].
        "image/filename": tf.string tensor indicating image filename.
        "caption/text": tf.string the string for the actual caption.
        "caption/ids": tf.int of IDs of the tokenized caption.
    """

    features = {
        # Images can have variable shape
        "image":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/filename":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "caption/text":
            tf.io.VarLenFeature(tf.string),
        "caption/embedding":
            tf.io.FixedLenFeature(self.embeddding_shape, tf.float32),
        "caption/max_len":
            tf.io.VarLenFeature(tf.int64),
    }

    decoded_example = tf.io.parse_single_example(example, features)

    decoded_example["image"] = tf.image.decode_png(
        decoded_example["image"], channels=3)
    decoded_example["image"] = tf.image.convert_image_dtype(
        decoded_example["image"], tf.float32)

    decoded_example["caption/max_len"] = tf.sparse.to_dense(
        decoded_example["caption/max_len"])
    decoded_example["caption/text"] = tf.sparse.to_dense(
        decoded_example["caption/text"], default_value="")

    return decoded_example

  def get_file_patterns(self,
                        split: Optional[str] = None,
                        file_pattern: Optional[str] = None):
    """Pass the file pattern to find corresponding files for given split."""

    if not file_pattern:
      if split not in ("train", "val"):
        raise ValueError(
            f"Expected split to be one of ['train', 'val'], got {split}")
      if split == "val":
        split = "validation"
      file_pattern = self.data_dir + (f"*{self.coco_version}*{split}.tfrecord*")
    return file_pattern

  def preprocess(self,
                 features: Dict[str, tf.Tensor],
                 training: bool = True) -> Dict[str, tf.Tensor]:
    rng = features.pop("rng")
    rng_flip, rng_sent_idx, rng_z, rng_aug = tf.unstack(
        tf.random.experimental.stateless_split(rng, 4))
    image = tf.image.resize(
        features["image"], (self.image_size, self.image_size),
        method="bilinear")
    image = tf.image.stateless_random_flip_left_right(image, rng_flip)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image_aug = augmentation.augment(image[None, ...], seed=rng_aug)[0, ...]
    image_filename = features["image/filename"]
    embedding = features["caption/embedding"]
    max_len = features["caption/max_len"]
    max_len = tf.dtypes.cast(tf.expand_dims(max_len, axis=-1), tf.float32)
    sentence_feat = tf.reduce_sum(embedding, axis=-2) / max_len
    idx = tf.random.stateless_uniform([],
                                      rng_sent_idx,
                                      minval=0,
                                      maxval=self.sentence_num,
                                      dtype=tf.int32)
    # If text should be returned, return the shortest caption rather than a
    # random one. This is the eval setup used in most text-to-image GANs.
    if self.return_text:
      idx = tf.argsort(
          features["caption/max_len"], axis=0, direction="DESCENDING")[-1]

    output = dict(
        image=tf.cast(image, self.data_dtype),
        image_aug=tf.cast(image_aug, self.data_dtype),
        image_filename=image_filename,
        embedding=tf.cast(embedding[idx], self.data_dtype),
        max_len=tf.cast(max_len[idx], self.data_dtype),
        sentence_embedding=tf.cast(sentence_feat[idx], self.data_dtype),
    )
    if self.return_text:
      output["text"] = features["caption/text"][idx]
    if self.return_filename:
      output["filename"] = features["image/filename"]
    z = tf.random.stateless_normal((self.z_dim,), rng_z, dtype=self.data_dtype)
    output.update({"z": z})
    return output

  @property
  def num_examples(self):
    if self.coco_version == "2017":
      return {"train": 116_680, "val": 4_958}
    elif self.coco_version == "2014":
      # return {"train": 82_783, "val": 40_504}
      return {"train": 1_000, "val": 200}
    elif self.coco_version == "ln":
      return {"train": 134_272, "val": 8_573}
    elif self.coco_version == "2014-tmage":
        return {"train": 1_000, "val": 200}
