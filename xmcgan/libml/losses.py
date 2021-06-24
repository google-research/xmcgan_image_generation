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

import jax
import jax.numpy as jnp


def hinge_loss_g(fake_logit: jnp.ndarray) -> jnp.ndarray:
  return -jnp.mean(fake_logit)


def hinge_loss_d(real_logit: jnp.ndarray,
                 fake_logit: jnp.ndarray) -> jnp.ndarray:
  real_loss = jnp.mean(jax.nn.relu(1.0 - real_logit))
  fake_loss = jnp.mean(jax.nn.relu(1.0 + fake_logit))
  return real_loss + fake_loss


def hinge_loss(real_logit: jnp.ndarray, fake_logit: jnp.ndarray) -> jnp.ndarray:
  generator_loss = -jnp.mean(fake_logit)
  real_loss = jax.nn.relu(1.0 - real_logit)
  fake_loss = jax.nn.relu(1.0 + fake_logit)
  discriminator_loss = jnp.mean(real_loss + fake_loss)
  return discriminator_loss, generator_loss


def cross_entropy_loss_with_logits(*, labels: jnp.ndarray,
                                   logits: jnp.ndarray) -> jnp.ndarray:
  """Calculates the cross entropy loss: label is one dimensional, not one hot."""

  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels[:, None], axis=1)
  return -loglik


def tf_cross_entropy_loss_with_logits(*, labels: jnp.ndarray,
                                      logits: jnp.ndarray) -> jnp.ndarray:
  logp = jax.nn.log_softmax(logits)
  loss = - jnp.sum(jnp.multiply(labels, logp), axis=-1)
  return loss

