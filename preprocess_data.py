# Copyright 2021 The XMC-GAN Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List

import bert
from bert import tokenization

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm

_CLS_TOKEN = '[CLS]'
_SEP_TOKEN = '[SEP]'

_BERT_LAYER = hub.KerasLayer(
    'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2')
_VOCAB_FILE = _BERT_LAYER.resolved_object.vocab_file.asset_path.numpy()
_DO_LOWER_CASE = _BERT_LAYER.resolved_object.do_lower_case.numpy()
_TOKENIZER = tokenization.FullTokenizer(_VOCAB_FILE, _DO_LOWER_CASE)


def get_bert_for_captions(captions: List[str], max_text_length: int = 17):
  """Returns BERT pooled and sequence outputs for a given list of captions."""
  all_tokens = []
  all_input_mask = []
  for text in captions:
    truncated_tokens = _TOKENIZER.tokenize(text)[:max_text_length - 2]
    tokens = [_CLS_TOKEN] + truncated_tokens + [_SEP_TOKEN]
    tokens = _TOKENIZER.convert_tokens_to_ids(tokens)
    num_padding = max_text_length - len(tokens)
    input_mask = [1] * len(tokens)
    tokens = tokens + [0] * num_padding
    input_mask = input_mask + [0] * num_padding
    all_tokens.append(np.asarray(tokens, np.int32))
    all_input_mask.append(np.asarray(input_mask, np.int32))

  ids = np.asarray(all_tokens)
  input_mask = np.asarray(all_input_mask)
  segment_ids = np.zeros_like(ids)  # Single segment input.

  _, embedding = _BERT_LAYER([ids, input_mask, segment_ids])
  max_len = np.sum(input_mask, axis=1)
  sent_embedding = tf.reduce_sum(embedding, axis=1) / max_len[:, None]
  return embedding, sent_embedding, max_len


def get_float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_bytes_feature(value):
  if not isinstance(value, list):
    value = [value]
  value_bytes = [tf.compat.as_bytes(element) for element in value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_bytes))


def serialize_example(ex):
  """Converts a TFDS coco_caption example into the expected format."""
  caption_list = [s.decode() for s in ex['captions']['text'].numpy()][:5]
  embedding, _, max_len = get_bert_for_captions(caption_list)
  image_data = tf.io.encode_png(ex['image'])
  filename = ex['image/filename']

  context_features = {
      'image':
          get_bytes_feature(image_data.numpy()),
      'image/filename':
          get_bytes_feature(filename.numpy()),
      'caption/embedding':
          get_float32_feature(embedding.numpy().flatten().tolist()),
      'caption/max_len':
          get_int64_feature(max_len),
      'caption/text':
          get_bytes_feature(caption_list),
  }
  tf_ex = tf.train.Example(features=tf.train.Features(feature=context_features))
  return tf_ex.SerializeToString()


if __name__ == '__main__':
  # Preprocess train and val data.
  for process_split in ['train', 'validation']:
    tfds_splits = ['train[:1000]']
    # COCO-2014 consists of 40k examples from these three splits.
    if process_split == 'validation':
      tfds_splits = ['restval[:200]', 'test[200:400]', 'val[400:600]']

    output_path = f'data/coco2014_{process_split}.tfrecord'
    with tf.io.TFRecordWriter(output_path) as file_writer:
      for tfds_split in tfds_splits:
        ds = tfds.load('coco_captions', split=tfds_split, data_dir='/ifs/loni/faculty/thompson/four_d/jnaik/cocodataset2014/data')
        logging.info(len(list(ds)))
        for features in tqdm(ds, position=0):
          file_writer.write(serialize_example(features))

    # Shard dataset.
    raw_dataset = tf.data.TFRecordDataset(output_path)
    num_shards = 10
    for i in range(num_shards):
      writer = tf.data.experimental.TFRecordWriter(
          f'{output_path}-{i}-of-{num_shards}')
      writer.write(raw_dataset.shard(num_shards, i))
    # Remove unsharded dataset.
    tf.io.gfile.remove(output_path)
