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

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.seed = 42

  config.eval_num = 30000
  config.eval_avg_num = 3
  config.num_train_steps = -1
  config.log_loss_every_steps = 1000
  config.eval_every_steps = 1000
  config.checkpoint_every_steps = 5000

  config.dataset = "mscoco"
  config.coco_version = "2014"
  config.data_dir = "data/"
  config.return_text = False
  config.return_filename = False

  config.trial = 0  # dummy for repeated runs.
  config.beta1 = 0.5
  config.beta2 = 0.999
  config.d_lr = 0.0004
  config.g_lr = 0.0001
  config.polyak_decay = 0.999
  config.show_num = 64
  config.shuffle_buffer_size = 1000
  config.batch_norm_group_size = -1
  config.dtype = "bfloat16"
  config.train_shuffle = True

  config.image_size = 128
  config.batch_size = 50
  config.eval_batch_size = 7

  config.df_dim = 96
  config.gf_dim = 96
  config.z_dim = 128
  config.num_epochs = 300
  config.model_name = "xmc"
  config.d_step_per_g_step = 2
  config.g_spectral_norm = False
  config.d_spectral_norm = True
  config.architecture = "xmc_net"
  config.gamma_for_g = 15
  config.word_contrastive = True
  config.sentence_contrastive = True
  config.image_contrastive = True
  config.pretrained_image_contrastive = True
  config.cond_size = 16

  return config


def get_test_config():
  """Get a smaller hyperparameter configuration for testing."""
  config = get_config()
  config.batch_size = 2
  config.eval_batch_size = 2
  config.eval_num = 2
  config.eval_avg_num = 1
  config.num_train_steps = 2
  config.log_loss_every_steps = 1
  config.eval_every_steps = 1
  config.checkpoint_every_steps = 1
  config.df_dim = 16
  config.gf_dim = 16
  config.z_dim = 8
  config.show_num = 4
  config.num_epochs = 1
  config.shuffle_buffer_size = 10
  return config


def get_hyper(h):
  return h.product([], name="config")
