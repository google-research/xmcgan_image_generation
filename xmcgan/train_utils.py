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
import os
from typing import Any, Dict, Sequence, Tuple, Union, Optional, List  # pylint: disable=unused-import
from absl import logging

from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from xmcgan import xmc_gan
from xmcgan.libml import input_pipeline
from xmcgan.nets import xmc_net
from xmcgan.utils import eval_metrics
from xmcgan.utils import image_utils
from xmcgan.utils import task_manager


@flax.struct.dataclass
class TrainState:
  """Data structure for checkpoint the model."""
  step: int
  g_optimizer: flax.optim.Optimizer
  d_optimizer: flax.optim.Optimizer
  generator_state: Optional[Any]
  discriminator_state: Optional[Any]
  ema_params: Any


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):

  eval_d_loss: metrics.Average.from_output("d_loss")
  eval_g_loss: metrics.Average.from_output("g_loss")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):

  d_loss: metrics.Average.from_output("d_loss")
  d_loss_std: metrics.Std.from_output("d_loss")
  g_loss: metrics.Average.from_output("g_loss")
  g_loss_std: metrics.Std.from_output("g_loss")


def split_input_dict(input_dict: Dict[str, jnp.ndarray], splits: int, axis=0):
  """Splits input dicts for multiple dicts for each step of training.

  Args:
    input_dict: Dict mapping keys to jnp.ndarray representing model inputs.
    splits: How many splits to separate the input_dict into.
    axis: Axis to perform the split. Defaults to 0 (batch dimension).
  Returns:
    output: List of individual input dictionaries.
  """

  output = []
  split_dict = jax.tree_map(lambda x: jnp.split(x, splits, axis=axis),
                            input_dict)
  for i in range(splits):
    current_dict = {}
    for key in input_dict.keys():
      current_dict[key] = split_dict[key][i]
    output.append(current_dict)
  return output


def train_step(
    rng: np.ndarray,
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    gan_model: Any,
    generator: Union[nn.Module, functools.partial],
    discriminator: Union[nn.Module, functools.partial],
    config: ml_collections.ConfigDict,
    additional_data: Dict[str, Any],
) -> Tuple[TrainState, metrics.Collection]:
  """Perform a single training step.

  Args:
    rng: The random seed,
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    gan_model: The GAN model defining the loss function.
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
  # Tmage1.0
  logging.info(f'batch {len(batch)}, {batch}')
  rngs = jax.random.split(rng, config.d_step_per_g_step)
  batch = split_input_dict(batch, config.d_step_per_g_step)

  for i in range(config.d_step_per_g_step - 1):
    state = gan_model.train_d(rngs[i], state, batch[i], generator,
                              discriminator, config)
  new_state, metrics_update = gan_model.train_g_d(rngs[-1], state, batch[-1],
                                                  generator, discriminator,
                                                  config, additional_data)
  return new_state, metrics_update


def create_train_state(
    config: ml_collections.ConfigDict, rng: np.ndarray,
    init_batch: Dict[str, np.ndarray]
) -> Tuple[Union[nn.Module, functools.partial], Union[
    nn.Module, functools.partial], TrainState]:
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    init_batch: The batch for initialization.

  Returns:
    The initialized TrainState with the optimizer.
  """
  if config.dtype == "bfloat16":
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32
  inputs = init_batch

  if config.architecture == "xmc_net":
    generator_cls = xmc_net.Generator
    discriminator_cls = xmc_net.Discriminator
  else:
    raise ValueError("Architecture {config.architecture} is not supported.")
  generator = functools.partial(generator_cls, config=config, dtype=dtype)
  discriminator = functools.partial(
      discriminator_cls, config=config, dtype=dtype)

  d_rng, g_rng, z_rng = jax.random.split(rng, 3)
  image = inputs["image"]
  batch_size = image.shape[0]
  # Tmage1.0
  logging.info(f"Batch size={batch_size}, image shape={image.shape}")
  z = jax.random.normal(z_rng, (batch_size, config.z_dim), dtype=dtype)
  generator_variables = generator(train=False).init(g_rng, (inputs, z))
  generator_state = dict(generator_variables)
  generator_params = generator_state.pop("params")
  ema_params = generator_params
  all_images = jnp.concatenate([image, image], axis=0)
  discriminator_variables = discriminator(train=False).init(
      d_rng, [all_images, inputs])
  discriminator_state = dict(discriminator_variables)
  discriminator_params = discriminator_state.pop("params")
  # Tmage1.0
  logging.info(f'{generator_state}, {generator_params}, {discriminator_state}, {discriminator_params}')

  logging.info("logging generator parameters")
  parameter_overview.log_parameter_overview(generator_params)
  logging.info("logging discriminator parameters")
  parameter_overview.log_parameter_overview(discriminator_params)
  g_optimizer = flax.optim.Adam(
      learning_rate=config.g_lr, beta1=config.beta1,
      beta2=config.beta2).create(generator_params)
  d_optimizer = flax.optim.Adam(
      learning_rate=config.d_lr, beta1=config.beta1,
      beta2=config.beta2).create(discriminator_params)
  return generator, discriminator, TrainState(
      step=0,
      g_optimizer=g_optimizer,
      d_optimizer=d_optimizer,
      generator_state=generator_state,
      discriminator_state=discriminator_state,
      ema_params=ema_params)


def generate_sample(
    rng: np.ndarray,
    state: TrainState,
    generator: Union[nn.Module, functools.partial],
    config: ml_collections.ConfigDict,
) -> Dict[str, jnp.ndarray]:
  """Generate sample in single device.

  Args:
    rng: PRNG key for pseudo random operations.
    state: TrainState object representing the model and parameters to use when
      sampling.
    generator: Function or Flax module representing generator object.
    config: Config dictionary used for sampling.
  Returns:
    output: Dictionary containing model generated image and generated image from
      model with exponential moving averaged weights.
  """

  if config.dtype == "bfloat16":
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32
  z_rng, label_rng = jax.random.split(rng, 2)
  sample_size = config.batch_size // jax.device_count()
  z = jax.random.normal(z_rng, (sample_size, config.z_dim), dtype=dtype)
  label = jax.random.randint(label_rng, (sample_size,), minval=0, maxval=1000)
  label = jax.nn.one_hot(label, config.num_classes)
  g_variables = {"params": state.g_optimizer.target}
  ema_g_variables = {"params": state.ema_params}
  g_variables.update(state.generator_state)
  ema_g_variables.update(state.generator_state)
  cond = dict(sentence_embedding=label)
  generated_image = generator(train=False).apply(
      g_variables, (cond, z), mutable=False)
  ema_generated_image = generator(train=False).apply(
      ema_g_variables, (cond, z), mutable=False)

  generated_image = image_utils.make_grid(generated_image, config.show_num)
  ema_generated_image = image_utils.make_grid(ema_generated_image,
                                              config.show_num)
  # Writer needs a 4D tensor
  generated_image = generated_image[None, :]
  ema_generated_image = ema_generated_image[None, :]

  return dict(
      generated_image=generated_image, ema_generated_image=ema_generated_image)


def generate_batch(rng: np.ndarray, state: TrainState, batch: Dict[str,
                                                                   jnp.ndarray],
                   generator: Union[nn.Module, functools.partial],
                   config: ml_collections.ConfigDict,
                   collect_all: bool = False) -> jnp.ndarray:
  """Generate batches.

  Args:
    rng: PRNG key for pseudo random operations.
    state: TrainState object representing the model and parameters to use when
      sampling.
    batch: Inputs to generate samples for.
    generator: Function or Flax module representing generator object.
    config: Config dictionary used for sampling.
    collect_all: If True, concatenates generation results across all replicas.
  Returns:
    output: Dictionary containing generated images, generated images from model
      with exponential moving averaged weights, and original images.
  """

  if config.dtype == "bfloat16":
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32

  # Tmage1.0
  logging.info(f'{len(batch)}')
  logging.info(f'{batch["image"].shape}')
  z = jax.random.normal(
      rng, (batch["image"].shape[0], config.z_dim), dtype=dtype)
  g_variables = {"params": state.g_optimizer.target}
  ema_g_variables = {"params": state.ema_params}
  g_variables.update(state.generator_state)
  ema_g_variables.update(state.generator_state)

  generated_image = generator(train=False).apply(
      g_variables, (batch, z), mutable=False)
  ema_generated_image = generator(train=False).apply(
      ema_g_variables, (batch, z), mutable=False)
  generated_image = jnp.asarray(generated_image, jnp.float32)
  ema_generated_image = jnp.asarray(ema_generated_image, jnp.float32)
  image_shape = list(generated_image.shape[-3:])
  image_shape = [-1] + image_shape
  ori_image = batch["image"]

  if collect_all:
    generated_image = jax.lax.all_gather(generated_image, axis_name="batch")
    ema_generated_image = jax.lax.all_gather(
        ema_generated_image, axis_name="batch")
    ori_image = jax.lax.all_gather(ori_image, axis_name="batch")

  generated_image = jnp.reshape(generated_image, image_shape)
  generated_image = image_utils.make_grid(generated_image, config.show_num)

  ema_generated_image = jnp.reshape(ema_generated_image, image_shape)
  ema_generated_image = image_utils.make_grid(ema_generated_image,
                                              config.show_num)

  ori_image = jnp.reshape(ori_image, image_shape)
  ori_image = image_utils.make_grid(ori_image, config.show_num)
  # Writer needs a 4D tensor
  generated_image = generated_image[None, :]
  ema_generated_image = ema_generated_image[None, :]
  ori_image = ori_image[None, :]

  return dict(
      generated_image_batch=generated_image,
      ema_generated_image_batch=ema_generated_image,
      ori_image_batch=ori_image)


def train(config: ml_collections.ConfigDict, workdir: str,
          test_mode: bool = False):
  """Runs a training loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    test_mode: If true, runs just one training iteration.
  """
  logging.info("Entered train_utils train method")
  tf.io.gfile.makedirs(workdir)
  rng = jax.random.PRNGKey(config.seed)
  
  if config.model_name == "xmc":
    gan_model = xmc_gan
  else:
    raise NotImplementedError(f"{config.model_name} was not Implemented!")
  additional_data = gan_model.create_additional_data(config)
  # Input pipeline.
  rng, data_rng = jax.random.split(rng)
  # Make sure each host uses a different RNG for the training data.
  logging.info("create dataset")
  data_rng = jax.random.fold_in(data_rng, jax.host_id())
  train_ds, eval_ds, num_train_examples = input_pipeline.create_datasets(
      config, data_rng)
  logging.info(train_ds)

  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

  num_train_steps = config.num_train_steps
  if num_train_steps == -1:
    if config.dataset == "mscoco":
      single_device_batch_size = config.batch_size / jax.device_count()
      steps_per_epoch = num_train_examples // (
          jax.local_device_count() * config.d_step_per_g_step)
      num_train_steps = steps_per_epoch * config.num_epochs
    else:
      num_train_steps = train_ds.cardinality().numpy()
      if test_mode:
        num_train_steps = 1
      steps_per_epoch = num_train_steps // config.num_epochs
    logging.info("total_epochs=%d, total_steps=%d, steps_per_epoch=%d",
                 config.num_epochs, num_train_steps, steps_per_epoch)
  else:
    logging.info("total_steps=%d", num_train_steps)

  # Initialize model
  logging.info("Model initalization")
  rng, model_rng = jax.random.split(rng)
  init_batch = jax.tree_map(np.asarray, next(train_iter))
  init_batch = jax.tree_map(
      lambda x: x[0], init_batch)  # Remove the device dim, still 4D tensor
  img = init_batch['image']
  logging.info(f'1 batch  {len(img)}')
  init_batch = split_input_dict(init_batch, config.d_step_per_g_step)
  init_batch = init_batch[0]

  generator, discriminator, state = create_train_state(config, model_rng,
                                                       init_batch)
  # Shape (#local_device, d_step_g_ste*per_replica_size, ...)
  batch_visualize = jax.tree_map(np.asarray, next(train_iter))
  batch_visualize = split_input_dict(
      batch_visualize, config.d_step_per_g_step, axis=1)[0]

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  task_manager_csv = task_manager.TaskManagerWithCsvResults(checkpoint_dir)
  ckpt = checkpoint.MultihostCheckpoint(
      checkpoint_dir, dict(train_iter=train_iter), max_to_keep=5)

  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1
  # Replicate our parameters.
  state = flax_utils.replicate(state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          gan_model=gan_model,
          generator=generator,
          discriminator=discriminator,
          config=config,
          additional_data=additional_data,
      ),
      axis_name="batch")
  p_generate_batch = jax.pmap(
      functools.partial(
          generate_batch,
          generator=generator,
          config=config,
      ),
      axis_name="batch")
  jitted_generated_sample = jax.jit(
      functools.partial(generate_sample, generator=generator, config=config))

  # Only write metrics on host 0, write to logs on all other hosts.
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.host_id() > 0)
  if initial_step == 1:
    writer.write_hparams(dict(config))
  logging.info("Starting training loop at step %d.", initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  if jax.host_id() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    ]
  train_metrics = None
  rng, train_rng, sample_rng, sample_batch_rng = jax.random.split(rng, 4)  # pylint: disable=unused-variable
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      logging.info(f'Current step: {step}')
      is_last_step = step == config.num_train_steps
      with jax.profiler.StepTraceContext("train", step_num=step):
        batch = jax.tree_map(np.asarray, next(train_iter))
        step_rng = jax.random.fold_in(train_rng, step)
        step_rngs = jax.random.split(step_rng, jax.local_device_count())
        state, metrics_update = p_train_step(step_rngs, state, batch)
        metric_update = flax.jax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)
      if (step % config.log_loss_every_steps == 0 or
          is_last_step) and config.dataset == "imagenet2012":
        logging.info("log step")
        single_state = flax.jax_utils.unreplicate(state)
        step_sample_rng = jax.random.fold_in(sample_rng, step)
        image_dict = jitted_generated_sample(step_sample_rng, single_state)
        image_dict = jax.tree_map(np.array, image_dict)
        writer.write_images(step, image_dict)
      if step % config.eval_every_steps == 0 or is_last_step:
        writer.write_scalars(step, train_metrics.compute())
        logging.info("eval step")
        batch_visualize = split_input_dict(
            batch, config.d_step_per_g_step, axis=1)[0]
        step_sample_batch_rng = jax.random.fold_in(sample_batch_rng, step)
        step_sample_batch_rngs = jax.random.split(step_sample_batch_rng,
                                                  jax.local_device_count())
        image_dict = p_generate_batch(step_sample_batch_rngs, state,
                                      batch_visualize)
        image_dict = flax.jax_utils.unreplicate(image_dict)
        image_dict = jax.tree_map(np.array, image_dict)
        writer.write_images(step, image_dict)
        train_metrics = None

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          ckpt.save(flax.jax_utils.unreplicate(state))
    logging.info("Finishing training at step %d", num_train_steps)
  task_manager_csv.mark_training_done()


def test(config: ml_collections.ConfigDict, workdir: str):
  """Runs a test run.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  rng = jax.random.PRNGKey(config.seed)
  # Make sure each host uses a different RNG for the training data.
  rng = jax.random.fold_in(rng, jax.host_id())
  # Input pipeline.
  rng, data_rng = jax.random.split(rng)
  _, eval_ds, _ = input_pipeline.create_datasets(config, data_rng)
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

  # Create evaluation metric utils and task manager.
  eval_metric = eval_metrics.EvalMetric(eval_iter, config)  # pytype: disable=wrong-arg-types
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  task_manager_csv = task_manager.TaskManagerWithCsvResults(checkpoint_dir)
  checkpoints = task_manager_csv.unevaluated_checkpoints(timeout=24 * 3600)
  writer = metric_writers.create_default_writer(workdir,
                                                just_logging=jax.host_id() > 0)

  rng, eval_rng, model_rng = jax.random.split(rng, 3)
  init_batch = jax.tree_map(np.asarray, next(eval_iter))
  init_batch = jax.tree_map(
      lambda x: x[0], init_batch)  # Remove the device dim, still 4D tensor
  generator, _, state = create_train_state(config, model_rng, init_batch)
  eval_generator = functools.partial(generator, train=False)

  with metric_writers.ensure_flushes(writer):
    for checkpoint_path in checkpoints:
      state = task_manager_csv.ckpt.restore_from_path(state, checkpoint_path)
      state = flax_utils.replicate(state)
      (fid, fid_std, inception_score, inception_score_std, ema_fid, ema_fid_std,
       ema_inception_score,
       ema_inception_score_std) = eval_metric.calculate_inception_fid(
           eval_generator, state, eval_rng)
      result_dict = dict(
          fid=fid,
          inception_score=inception_score,
          fid_std=fid_std,
          inception_score_std=inception_score_std,
          ema_fid=ema_fid,
          ema_inception_score=ema_inception_score,
          ema_fid_std=ema_fid_std,
          ema_inception_score_std=ema_inception_score_std)
      result_dict = {f"eval/{k}": v for k, v in result_dict.items()}
      task_manager_csv.add_eval_result(checkpoint_path, result_dict, -1)
      writer.write_scalars(np.array(state.step)[0], result_dict)
