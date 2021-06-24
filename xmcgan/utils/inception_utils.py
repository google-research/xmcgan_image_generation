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

import re
from typing import Any, Dict, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from xmcgan.utils import inception_arch
from xmcgan.utils import tf_inception_utils


def map_keras_variables_to_flax_dict(
    keras_model: tf.keras.Model,
    include_top: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Map the weights from a TensorFlow Keras model to a Flax friendly format.

  Args:
    keras_model: an Inception V3 model instantiated in Keras. If include_top is
      true the last layer must be present.
    include_top: if True extract the weights for the last dense layer too.

  Returns:
    A pair of nested pytrees: for the first the leaves are the model weights,
      for the second the leaves are the batch stats for the batch norm layers.
  """
  # Holds weights as numpy arrays.
  jvars = {}
  # Holds batch stats as numpy arrays.
  jstats = {}

  # Dictionary helping converting variable names from Keras to JaX.
  keras_to_jax_names = {
      'conv2d': 'Conv_0',
      'batch_normalization': 'BatchNorm_0',
      'kernel:0': 'kernel',
      'beta:0': 'bias',
      'moving_mean:0': 'mean',
      'moving_variance:0': 'var'
  }

  # Pass 1: Populate all the ConvBatchNormReluBlocks.
  # In this way all the nested dictionaries have the keys ready.
  for v in keras_model.variables:
    matches = re.match('conv2d(.*)/kernel:0', v.name)
    if matches:
      num = matches.groups()[0]
      if not num:
        num = '_0'  # Keras has 0 layers without a _0.
      layer_key = 'ConvBatchNormReluBlock' + num
      jvars[layer_key] = {'Conv_0': {}, 'BatchNorm_0': {}}
      jstats[layer_key] = {'BatchNorm_0': {}}

  # Pass 2 populate variables for the blocks.
  for v in keras_model.variables:
    matches = re.match('(conv2d|batch_normalization)(.*)', v.name)
    if matches:
      sub_matches = re.match('(conv2d|batch_normalization)(.*)/(.*)', v.name)
      if sub_matches:
        matched = sub_matches.groups()
        layer_type, num, param = matched[0], matched[1], matched[2]
        if not num:
          num = '_0'
        layer_key = 'ConvBatchNormReluBlock' + num
        layer_type = keras_to_jax_names[layer_type]
        param = keras_to_jax_names[param]
        if layer_type == 'Conv_0':
          jvars[layer_key][layer_type]['kernel'] = v.numpy()
        elif layer_type == 'BatchNorm_0' and param == 'bias':
          jvars[layer_key][layer_type]['bias'] = v.numpy()
          jvars[layer_key][layer_type]['scale'] = np.ones_like(
              jvars[layer_key][layer_type]['bias'])
        elif layer_type == 'BatchNorm_0' and param in ['mean', 'var']:
          jstats[layer_key][layer_type][param] = v.numpy()

  if include_top:
    top = keras_model.get_layer('predictions')
    jvars['Dense_0'] = {'kernel': top.kernel.numpy(), 'bias': top.bias.numpy()}
  return jvars, jstats


def get_inception(image,
                  model: flax.linen.Module,
                  model_params: flax.core.FrozenDict,
                  resize_mode: str = 'bilinear',
                  re_normalize: bool = True):
  """Returns Inception model pools and logits.

  Args:
    image: (N, num_replica, H, W, 3) array object for images in the batch.
    model: A Flax Module for the Inception model to be used.
    model_params: Flax dictionary containing model parameters.
    resize_mode: A string for indicating the resize method.
    re_normalize: True to renorm the image range from [0, 1] to [-1, 1].

  Returns:
    pool: (N, 2048) array of Inception v3 pool values.
    preds: (N, 1000) array of Inception v3 probability values.
  """
  if image.shape[1] != 299 or image.shape[2] != 299:
    image = jax.image.resize(image, [image.shape[0], 299, 299, image.shape[3]],
                             resize_mode)
  # Inception model accepts image from [-1, 1], if images are from [0, 1],
  # scale it to [-1, 1]
  if re_normalize:
    image = jnp.clip(image * 2 - 1., -1., 1.)

  pool, logits = model.apply(model_params, image)
  preds = nn.softmax(logits)
  return pool, preds


def inception_model(checkpoint_path: Optional[str] = None):
  """Inception Model loading function.

  Args:
    checkpoint_path: Path to load the model from. This should be a
      tf.keras.Model checkpoint for the Inception v3 model.

  Returns:
    jax_model: Inception v3 Flax module.
    model_params: Dictionary containing Flax parameters.
  """
  tf_model = tf_inception_utils.inception_model(checkpoint_path=checkpoint_path)
  # Init the model with a random image
  jax_model = inception_arch.InceptionV3(include_top=True)
  # Extract the weights
  jvars, jstats = map_keras_variables_to_flax_dict(tf_model)
  j_params = flax.core.FrozenDict(jvars)
  model_params = flax.core.FrozenDict({
      'params': j_params,
      'batch_stats': jstats
  })
  return jax_model, model_params


def calculate_fid(pool1, pool2):
  """Numpy version of FID calculation.

  Full function defined in
  //learning/brain/research/ruml/gan/text2img/utils/inception_utils.py.

  Args:
    pool1: A numpy array for the pooling features of first instance.
    pool2: A numpy array for the pooling features of second instance.

  Returns:
    fid_value: A numpy array or a float value for the FID.
  """
  return tf_inception_utils.calculate_fid(pool1, pool2)


def calculate_inception_score(pred, num_splits=10):
  """Numpy version of Inception Score calculation.

  Full function defined in
  //learning/brain/research/ruml/gan/text2img/utils/inception_utils.py.

  Args:
    pred: A numpy array for the logits.
    num_splits: The number of splits for calculating inception score.

  Returns:
    The average and standard deviation of the calculated inception score.
  """
  return tf_inception_utils.calculate_inception_score(pred, num_splits)
