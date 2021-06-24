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

import csv
import os
import time
from typing import Any, Dict, Iterable, Optional, TypeVar

from absl import logging
from clu import checkpoint as checkpoint_lib
import flax
import tensorflow as tf

T = TypeVar("T")


class QueryMultihostCheckpoint(checkpoint_lib.MultihostCheckpoint):
  """An subclass of `Checkpoint` that synchronizes between multiple JAX hosts.
  """

  def get_all_checkpoints_to_restore_from(self):
    """Returns the latest checkpoint available on all hosts."""
    base_directory_glob = f"{self.multihost_base_directory}-*"
    base_directories = tf.io.gfile.glob(base_directory_glob)
    if self.base_directory not in base_directories:
      return None
    checkpoints = {}
    for base_directory in base_directories:
      checkpoint_manager = tf.train.CheckpointManager(
          tf.train.Checkpoint(),
          base_directory,
          max_to_keep=self.max_to_keep,
          checkpoint_name=self.checkpoint_name)
      checkpoints[base_directory] = set(checkpoint_manager.checkpoints)
    if not checkpoints:
      return {}
    return checkpoints[self.base_directory]

  def restore_from_path(self, state: T, path: str) -> T:
    """Restores from a given checkpoint path.

    Args:
      state : A flax checkpoint to be stored or to serve as a template. If the
        checkoint is restored (and not initialized), then the fields of `state`
        must match the data previously stored.
      path: Path to checkpoint to restore.

    Returns:
      The restored `state` object. Note that all TensorFlow `Trackable`s in
      `tf_state` (see `__init__()`) are also updated.
    """
    self.tf_checkpoint.restore(path)
    flax_path = self._flax_path(path)
    with tf.io.gfile.GFile(flax_path, "rb") as f:
      state = flax.serialization.from_bytes(state, f.read())
    return state


class TaskManager:
  """Class for checking the model folder repeately for evaluation."""

  def __init__(self, model_dir: str) -> None:
    self.ckpt = QueryMultihostCheckpoint(model_dir, {})
    self._model_dir = self.ckpt.base_directory

  @property
  def model_dir(self) -> str:
    return self._model_dir

  def mark_training_done(self) -> None:
    with tf.io.gfile.GFile(
        os.path.join(self.model_dir, "TRAIN_DONE"), "w") as f:
      f.write("")

  def is_training_done(self) -> None:
    return tf.io.gfile.exists(os.path.join(self.model_dir, "TRAIN_DONE"))

  def add_eval_result(
      self,
      checkpoint_path: str,
      result_dict: Dict[str, Any],
      default_value: int = -1) -> None:
    pass

  def _get_checkpoints_with_results(self):
    return set()

  def unevaluated_checkpoints(self,
                              timeout: int = 3600 * 8,
                              num_batched_steps: int = 1,
                              eval_every_steps: Optional[int] = None,
                              ) -> Iterable[str]:
    """Generator for checkpoints without evaluation results.

    Args:
      timeout: Optional timeout for waiting for new checkpoints. Set this to
        do continious evaluation.
      num_batched_steps: Steps that are batched into a single tf.function.
        Required for computing correct evaluation checkpoints.
      eval_every_steps: Only evaluate checkpoints from steps divisible by this
                         integer.

    Yields:
      Path to checkpoints that have not yet been evaluated.
    """
    logging.info("Looking for checkpoints in %s", self.ckpt.base_directory)
    evaluated_checkpoints = self._get_checkpoints_with_results()
    last_eval = time.time()
    while True:
      # Check if directory exists. The train job may only create the directory
      # some time after the test job starts.
      if not tf.io.gfile.exists(self.ckpt.base_directory):
        logging.info("Directory %s does not exist!", self.ckpt.base_directory)
      else:
        logging.info("what is in %s:  are  %s", self.ckpt.base_directory,
                     tf.io.gfile.listdir(self.ckpt.base_directory))
        unevaluated_checkpoints = []
        checkpoints = self.ckpt.get_all_checkpoints_to_restore_from()
        logging.info("checkpoints: %s", checkpoints)
        unevaluated_checkpoints = checkpoints - evaluated_checkpoints
        step_and_ckpt = sorted(
            (int(x.split("-")[-1]), x) for x in unevaluated_checkpoints)

        unevaluated_checkpoints = []
        for step, ckpt in step_and_ckpt:
          if eval_every_steps:
            if step > num_batched_steps and (
                step % eval_every_steps < num_batched_steps):
              unevaluated_checkpoints.append(ckpt)
          else:
            unevaluated_checkpoints.append(ckpt)

        logging.info(
            "Found checkpoints: %s\nEvaluated checkpoints: %s\n"
            "Unevaluated checkpoints: %s", checkpoints, evaluated_checkpoints,
            unevaluated_checkpoints)
        for checkpoint_path in unevaluated_checkpoints:
          yield checkpoint_path

        if unevaluated_checkpoints:
          evaluated_checkpoints |= set(unevaluated_checkpoints)
          last_eval = time.time()
          continue
      if time.time() - last_eval > timeout or self.is_training_done():
        break
      time.sleep(5)


class TaskManagerWithCsvResults(TaskManager):
  """Task Manager that writes results to a CSV file."""

  def __init__(self,
               model_dir: str,
               score_file: Optional[str] = None) -> None:
    super().__init__(model_dir)
    if score_file is None:
      score_file = os.path.join(self._model_dir, "scores.csv")
    else:
      score_file = os.path.join(self._model_dir, score_file)
    self._score_file = score_file

  def _get_checkpoints_with_results(self):
    """Return the checkpoints as set."""
    if not tf.io.gfile.exists(self._score_file):
      return set()
    with tf.io.gfile.GFile(self._score_file) as f:
      reader = csv.DictReader(f)
      return {r["checkpoint_path"] for r in reader}
    return set()

  def add_eval_result(self,
                      checkpoint_path: str,
                      result_dict: Dict[str, Any],
                      default_value: int) -> None:
    """Add eval result to the CSV file."""
    step = int(os.path.basename(checkpoint_path).split("-")[-1])
    csv_header = (
        ["checkpoint_path", "step"] + sorted(result_dict))
    write_header = not tf.io.gfile.exists(self._score_file)
    if write_header:
      with tf.io.gfile.GFile(self._score_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction="ignore")
        writer.writeheader()
    row = dict(checkpoint_path=checkpoint_path, step=str(step))
    for k, v in result_dict.items():
      if isinstance(v, float):
        v = "{:.3f}".format(v)
      row[k] = v
    with tf.io.gfile.GFile(self._score_file, "a") as f:
      writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction="ignore")
      writer.writerow(row)
