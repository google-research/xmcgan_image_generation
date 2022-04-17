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
from typing import Any, Dict, Sequence, Tuple, Union, Optional, List  # pylint: disable=unused-import
from absl import logging

from clu import metrics

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from xmcgan.libml import attention_lib as attn_lib  # pylint: disable=unused-import
from xmcgan.libml import losses
from xmcgan.utils import pretrained_model_utils


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):

  d_loss: metrics.Average.from_output("d_loss")
  g_loss: metrics.Average.from_output("g_loss")
  c_loss_d: metrics.Average.from_output("c_loss_d")
  c_loss_g: metrics.Average.from_output("c_loss_g")
  c_loss_g_pretrained: metrics.Average.from_output("c_loss_g_pretrained")


def create_additional_data(config: ml_collections.ConfigDict):
  """Returns additional data required to run the model."""
  image_model = None
  image_model_state = None
  additional_data = {}
  if config.pretrained_image_contrastive:
    (image_model,
     image_model_state) = pretrained_model_utils.get_pretrained_model()
    additional_data.update({
        "image_model": image_model,
        "image_model_state": image_model_state,
    })
  return additional_data


def calculate_contrastive_loss(result_dict):
  """Calculates contrastive loss.

  Args:
    result_dict: Dictionary output from the discriminator.
  Returns:
    c_loss_d: Contrastive loss for the discriminator.
    c_loss_g: Contrastive loss for the generator.
  """
  real_loss = result_dict["real_word_loss"] + result_dict["real_sentence_loss"]
  fake_loss = result_dict["fake_word_loss"] + result_dict["fake_sentence_loss"]
  c_loss_d = real_loss
  c_loss_g = fake_loss + result_dict["image_contrastive_loss"]
  return c_loss_d, c_loss_g


def calculate_contrastive_loss_on_pretrained(model: nn.Module, state: Any,
                                             real_images: jnp.ndarray,
                                             fake_images: jnp.ndarray):
  """Calculates contrastive loss on pre-trained model.

  Args:
    model: Pretrained model used for computing features.
    state: TrainState object at current training step.
    real_images: Array of real images.
    fake_images: Array of generated images.
  """
  _, real_outputs = pretrained_model_utils.get_pretrained_embs(
      state, model, images=real_images)
  _, fake_outputs = pretrained_model_utils.get_pretrained_embs(
      state, model, images=fake_images)
  loss, _, _ = attn_lib.contrastive_loss(real_outputs, fake_outputs)
  return loss


def train_g_d(
    rng: np.ndarray,
    state: Any,
    batch: Dict[str, jnp.ndarray],
    generator: Union[nn.Module, functools.partial],
    discriminator: Union[nn.Module, functools.partial],
    config: ml_collections.ConfigDict,
    additional_data: Dict[str, Any]) -> Tuple[Any, metrics.Collection]:
  """Perform a single training step.

  Args:
    rng: The random seed,
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    generator: Flax module for the generator. The apply method must take input
      images and a boolean argument indicating whether to use training or
      inference mode.
    discriminator: Flax module for the discriminator. The apply method must take
      input images and a boolean argument indicating whether to use training or
      inference mode.
    config: Configuration for model.
    additional_data: Dictionary containing model specific data / networks.

  Returns:
    The new model state and dictionary with metrics
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1
  if config.dtype == "bfloat16":
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32

  def loss_fn(params_d, params_g):
    g_variables = {"params": params_g}
    g_variables.update(state.generator_state)
    d_variables = {"params": params_d}
    d_variables.update(state.discriminator_state)
    if "z" in batch:
      z = batch["z"]
    else:
      z = jax.random.normal(
          rng, (batch["image"].shape[0], config.z_dim), dtype=dtype)
    real_image = batch["image"]
    generated_image, new_g_variables = generator(train=True).apply(
        g_variables, (batch, z), mutable=["batch_stats", "spectral_norm_stats"])
    all_images = jnp.concatenate([real_image, generated_image])
    (logit, result_dict), new_d_variables = discriminator(train=True).apply(
        d_variables, (all_images, batch),
        mutable=["batch_stats", "spectral_norm_stats"])
    logit = jnp.asarray(logit, jnp.float32)
    real_logit, fake_logit = jnp.split(logit, 2)
    d_loss, g_loss = losses.hinge_loss(real_logit, fake_logit)
    c_loss_d, c_loss_g = calculate_contrastive_loss(result_dict)
    c_loss_g_pretrained = 0.0
    if config.pretrained_image_contrastive:
      c_loss_g_pretrained = calculate_contrastive_loss_on_pretrained(
          additional_data["image_model"], additional_data["image_model_state"],
          real_image, generated_image)
    d_loss = d_loss + c_loss_d
    g_loss = g_loss + c_loss_g + c_loss_g_pretrained
    new_g_state = dict(new_g_variables)
    new_d_state = dict(new_d_variables)
    return (d_loss, g_loss), (new_g_state, new_d_state, c_loss_d, c_loss_g,
                              c_loss_g_pretrained)

  params_d = state.d_optimizer.target
  params_g = state.g_optimizer.target
  (d_loss, g_loss), func_vjp, (new_g_state, new_d_state, c_loss_d, c_loss_g,
                               c_loss_g_pretrained) = jax.vjp(
                                   loss_fn, params_d, params_g, has_aux=True)

  d_grad, _ = func_vjp((1., 0.))
  _, g_grad = func_vjp((0., 1.))

  # Compute average gradient across multiple workers.
  d_grad = jax.lax.pmean(d_grad, axis_name="batch")
  g_grad = jax.lax.pmean(g_grad, axis_name="batch")
  new_d_optimizer = state.d_optimizer.apply_gradient(d_grad)
  new_g_optimizer = state.g_optimizer.apply_gradient(g_grad)
  ema_decay = config.polyak_decay
  new_ema_params = jax.tree_multimap(
      lambda ema, p: ema * ema_decay + (1 - ema_decay) * p, state.ema_params,
      new_g_optimizer.target)
  new_state = state.replace(  # pytype: disable=attribute-error
      step=step,
      d_optimizer=new_d_optimizer,
      g_optimizer=new_g_optimizer,
      generator_state=new_g_state,
      discriminator_state=new_d_state,
      ema_params=new_ema_params)
  metrics_update = TrainMetrics.gather_from_model_output(
      g_loss=g_loss,
      d_loss=d_loss,
      c_loss_d=c_loss_d,
      c_loss_g=c_loss_g,
      c_loss_g_pretrained=c_loss_g_pretrained)
  return new_state, metrics_update


def train_d(rng: np.ndarray, state: Any, batch: Dict[str, jnp.ndarray],
            generator: Union[nn.Module, functools.partial],
            discriminator: Union[nn.Module, functools.partial],
            config: ml_collections.ConfigDict) -> Any:
  """Perform a single training step.

  Args:
    rng: The random seed,
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    generator: Flax module for the generator. The apply method must take input
      images and a boolean argument indicating whether to use training or
      inference mode.
    discriminator: Flax module for the discriminator. The apply method must take
      input images and a boolean argument indicating whether to use training or
      inference mode.
    config: Configuration for model.

  Returns:
    The new model state.
  """
  if config.dtype == "bfloat16":
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32

  def loss_fn(params_d, params_g):
    g_variables = {"params": params_g}
    g_variables.update(state.generator_state)
    d_variables = {"params": params_d}
    d_variables.update(state.discriminator_state)
    if "z" in batch:
      z = batch["z"]
    else:
      z = jax.random.normal(
          rng, (batch["image"].shape[0], config.z_dim), dtype=dtype)

    generated_image, _ = generator(train=True).apply(
        g_variables, (batch, z), mutable=["batch_stats", "spectral_norm_stats"])
    all_images = jnp.concatenate([batch["image"], generated_image])
    (logit, result_dict), new_d_variables = discriminator(train=True).apply(
        d_variables, (all_images, batch),
        mutable=["batch_stats", "spectral_norm_stats"])
    logit = jnp.asarray(logit, jnp.float32)
    real_logit, fake_logit = jnp.split(logit, 2)
    d_loss, _ = losses.hinge_loss(real_logit, fake_logit)
    c_loss_d, _ = calculate_contrastive_loss(result_dict)
    d_loss += c_loss_d
    new_d_state = dict(new_d_variables)
    return d_loss, new_d_state

  params_d = state.d_optimizer.target
  params_g = state.g_optimizer.target
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (d_loss, new_d_state), d_grad = grad_fn(params_d, params_g)
  del d_loss
  # Compute average gradient across multiple workers.
  d_grad = jax.lax.pmean(d_grad, axis_name="batch")
  new_d_optimizer = state.d_optimizer.apply_gradient(d_grad)
  new_state = state.replace(  # pytype: disable=attribute-error
      d_optimizer=new_d_optimizer,
      discriminator_state=new_d_state)
  return new_state
