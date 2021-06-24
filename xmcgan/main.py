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

from absl import app
from absl import flags
from absl import logging

from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf


from xmcgan import train_utils


FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config", "workdir"])
flags.DEFINE_enum("mode", "train", ["train", "test"], "job status")
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def main(argv):
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = ("None" if FLAGS.jax_xla_backend is None else
                       FLAGS.jax_xla_backend)
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  logging.info("JAX host: %d / %d", jax.host_id(), jax.host_count())
  logging.info("JAX devices: %r", jax.devices())

  # Add a note so that we can tell which Borg task is which JAX host.
  # (Borg task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"host_id: {jax.host_id()}, host_count: {jax.host_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")
  if FLAGS.mode == "train":
    train_utils.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "test":
    train_utils.test(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
