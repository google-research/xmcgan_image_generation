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

from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
import flax.linen as nn
from flax.linen.initializers import lecun_normal, zeros, normal  # pylint: disable=g-multiple-import
import jax
from jax import lax

import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any

default_kernel_init = lecun_normal()


def _l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.

  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class SpectralDense(nn.Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    train: whether training or testing.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    eps: The constant used for numerical stability. The default is 1e-10, same
      as TF version, but different from Haiku version.
  """
  features: int
  train: bool
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  eps: float = 1e-10

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel', self.kernel_init,
                        (inputs.shape[-1], self.features))
    u0_variable = self.variable(
        'spectral_norm_stats',
        'u0',
        lambda s: normal()  # pylint: disable=g-long-lambda
        (self.make_rng('params'), s),
        (1, self.features))
    u0 = u0_variable.value
    # One step of power iteration.
    v0 = _l2_normalize(jnp.matmul(u0, kernel.transpose([1, 0])), eps=self.eps)
    u0 = _l2_normalize(jnp.matmul(v0, kernel), eps=self.eps)
    u0 = jax.lax.stop_gradient(u0)
    v0 = jax.lax.stop_gradient(v0)
    if self.train:
      u0_variable.value = u0
    sigma = jnp.matmul(jnp.matmul(v0, kernel), jnp.transpose(u0))[0, 0]
    kernel = kernel / (sigma + self.eps)  # Different from Haiku version.

    kernel = jnp.asarray(kernel, self.dtype)

    y = lax.dot_general(
        inputs,
        kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class SpectralConv(nn.Module):
  """Convolution Module wrapping lax.conv_general_dilated.

  Attributes:
    features: number of convolution filters.
    train: Whether training or testing.
    kernel_size: shape of the convolutional kernel. For 1D convolution, the
      kernel size can be passed as an integer. For all other cases, it must be a
      sequence of integers.
    strides: a sequence of `n` integers, representing the inter-window strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    input_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of `inputs`. Convolution with
      input dilation `d` is equivalent to transposed convolution with stride
      `d`.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of the convolution kernel.
      Convolution with kernel dilation is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    eps: The constant used for numerical stability. The default is 1e-10, same
      as TF version, but different from Haiku version.
  """
  features: int
  train: bool
  kernel_size: Union[int, Iterable[int]]
  strides: Optional[Iterable[int]] = None
  padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
  input_dilation: Optional[Iterable[int]] = None
  kernel_dilation: Optional[Iterable[int]] = None
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: Dtype = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  eps: float = 1e-10

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).

    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, self.dtype)

    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)

    strides = self.strides or (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    kernel_shape = kernel_size + (in_features // self.feature_group_count,
                                  self.features)
    kernel = self.param('kernel', self.kernel_init, kernel_shape)
    kernel = jnp.reshape(kernel, [-1, self.features])

    u0_variable = self.variable(
        'spectral_norm_stats',
        'u0',
        lambda s: normal()  # pylint: disable=g-long-lambda
        (self.make_rng('params'), s),
        (1, self.features))
    u0 = u0_variable.value
    # One step of power iteration.
    v0 = _l2_normalize(jnp.matmul(u0, kernel.transpose([1, 0])), eps=self.eps)
    u0 = _l2_normalize(jnp.matmul(v0, kernel), eps=self.eps)
    u0 = jax.lax.stop_gradient(u0)
    v0 = jax.lax.stop_gradient(v0)
    if self.train:
      u0_variable.value = u0

    sigma = jnp.matmul(jnp.matmul(v0, kernel), jnp.transpose(u0))[0, 0]
    kernel = kernel / (sigma + self.eps)  # Different from Haiku version.
    kernel = jnp.reshape(kernel, kernel_shape)
    kernel = jnp.asarray(kernel, self.dtype)

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        self.padding,
        lhs_dilation=self.input_dilation,
        rhs_dilation=self.kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        precision=self.precision)

    if is_single_input:
      y = jnp.squeeze(y, axis=0)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


class ConditionalBatchNorm(nn.Module):
  """Condtional batch normalization layer."""
  norm_fn: Any
  dense_fn: Any

  @nn.compact
  def __call__(self, x, emb):
    filters = x.shape[-1]
    gamma = self.dense_fn(filters)(emb)
    gamma = jnp.reshape(gamma, (-1, 1, 1, filters))
    beta = self.dense_fn(filters)(emb)
    beta = jnp.reshape(beta, (-1, 1, 1, filters))
    x = self.norm_fn(use_bias=False, use_scale=False)(x)
    x = x * (gamma + 1.0) + beta
    return x


class LocalConditionalBatchNorm(nn.Module):
  """Condtional batch normalization layer."""
  norm_fn: Any
  conv_fn: Any

  @nn.compact
  def __call__(self, x, emb):
    filters = x.shape[-1]
    gamma = self.conv_fn(filters, kernel_size=(1, 1))(emb)
    beta = self.conv_fn(filters, kernel_size=(1, 1))(emb)
    x = self.norm_fn(use_bias=False, use_scale=False)(x)
    x = x * (gamma + 1.0) + beta
    return x
