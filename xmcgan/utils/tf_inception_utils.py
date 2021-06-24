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

from typing import Optional
import warnings
import gin
import numpy as np
from scipy import linalg
import tensorflow as tf


_INCEPTION_INPUT = "Mul:0"
_INCEPTION_OUTPUT = "logits:0"
_INCEPTION_FINAL_POOL = "pool_3:0"


class ShapeNotMatchError(Exception):
  """Prints error when the shape of two tensor does not match."""
  pass


class ImaginaryComponentError(Exception):
  """Prints error when the input has imaginary component."""
  pass


def _get_graph_def_from_file(filename):
  with tf.io.gfile.GFile(filename, "rb") as f:
    proto_str = f.read()
  return tf.compat.v1.GraphDef.FromString(proto_str)


class _InceptionLayer(tf.keras.layers.Layer):
  """Auxiliary class for loding Inception V1 weights."""

  def __init__(self, inception_path):

    def import_graph():
      my_graph = _get_graph_def_from_file(inception_path)
      tf.compat.v1.import_graph_def(my_graph, name="")

    wrapped = tf.compat.v1.wrap_function(import_graph, signature=[])
    self._call_fn = wrapped.prune(_INCEPTION_INPUT,
                                  [_INCEPTION_FINAL_POOL, _INCEPTION_OUTPUT])

  def __call__(self, inputs):
    pools, logits = self._call_fn(inputs)
    preds = tf.nn.softmax(logits)
    pools = tf.squeeze(pools)
    return pools, preds


def _inception_model_v1(checkpoint_path):
  """Load inception V1 Model."""
  return _InceptionLayer(checkpoint_path)


@gin.configurable
def inception_model(checkpoint_path: Optional[str] = None):
  """Inception Model loading function.

  Args:
    checkpoint_path: Path to load the model from. This is either a .pb file for
      Inception v1, or a tf.keras.Model checkpoint for Inception v3.
  Returns:
    An tf.keras.Model.
  """
  if checkpoint_path:
    inception_v3 = tf.keras.applications.InceptionV3(
        weights=None, input_shape=(299, 299, 3), include_top=True)
    # The model is loaded from a checkpoint to avoid memory issues on Borg that
    # arise from trying to download weights.
    ckpt = tf.train.Checkpoint(inception_v3=inception_v3)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
  else:
    inception_v3 = tf.keras.applications.InceptionV3(
        weights="imagenet", input_shape=(299, 299, 3), include_top=True)
  model = tf.keras.Model(
      inputs=inception_v3.input,
      outputs=[
          inception_v3.get_layer("avg_pool").output,
          inception_v3.get_layer("predictions").output
      ])
  model.trainable = False
  return model


def get_inception(image,
                  model,
                  resize_mode="bilinear",
                  re_normalize: bool = True):
  """Returns Inception model pools and logits.

  The pixel range is expected from [0, 1] by default.

  Args:
    image: An tf.Tensor object for images in the current batch.
    model: An tf.Module object for the inception model in the calculation.
    resize_mode: An string for indication the resize method.
    re_normalize: True to renorm the image range from [0, 1] to [-1, 1].
  """
  if image.shape[1] != 299 or image.shape[2] != 299:
    image = tf.image.resize(image, [299, 299], resize_mode)
  # Inception model accepts image from [-1, 1], if images are from [0, 1],
  # scale it to [-1, 1]
  if re_normalize:
    image = tf.clip_by_value(image * 2 - 1., -1., 1.)
  pools, logits = model(image)
  return pools, logits


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
      d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

  Args:
    mu1: Numpy array containing the activations of the pool_3 layer of the
      inception net ( like returned by the function "get_predictions")
      for generated samples.
    sigma1: The covariance matrix over activations of the pool_3 layer for
      generated samples.
    mu2: The sample mean over activations of the pool_3 layer,
      precalcualted on an representive data set.
    sigma2: The covariance matrix over activations of the pool_3 layer,
      precalcualted on an representive data set.
    eps: a lower bound for the diagnoal

  Returns:
    fid: The Frechet Distance.

  Raises:
    ShapeNotMatchError: An Error occurred when the shape of mu1 and mu2 do not
      match, or the shape of sigma1 and sigma2 do not mathch.
    ImaginaryComponentError: An Error occurred when numerical error
      might give slight imaginary component.
  """
  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  if mu1.shape != mu2.shape:
    raise ShapeNotMatchError("Training and test mean vectors have "
                             "different lengths")
  if sigma1.shape != sigma2.shape:
    raise ShapeNotMatchError("Training and test covariances have "
                             "different dimensions")

  diff = mu1 - mu2

  # product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = ("fid calculation produces singular product; adding %s to diagonal of"
           " cov estimates") % eps
    warnings.warn(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ImaginaryComponentError("Imaginary component {}".format(m))
    covmean = covmean.real

  tr_covmean = np.trace(covmean)
  fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  return fid


def calculate_fid(pool1, pool2):
  """Numpy version of FID calculation.

  Args:
    pool1: A numpy array for the pooling features of first instance.
    pool2: A numpy array for the pooling features of second instance.

  Returns:
    fid_value: A numpy array or a float value for the FID.
  """

  mu1 = np.mean(pool1, axis=0)
  mu2 = np.mean(pool2, axis=0)
  sigma1 = np.cov(pool1, rowvar=False)
  sigma2 = np.cov(pool2, rowvar=False)
  fid_value = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
  return fid_value


def calculate_inception_score(pred, num_splits=10):
  """Numpy version of Inception Score calculation.

  Args:
    pred: A numpy array for the logits.
    num_splits: The number of splits for calulating inception score.

  Returns:
    The avarge and standard deviation of the calculated inception score.
  """
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits):(index + 1) *
                      (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (
        np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)
