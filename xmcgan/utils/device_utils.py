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


def get_device_groups(group_batch_size, device_batch_size):
  assert group_batch_size % device_batch_size == 0
  group_size = group_batch_size // device_batch_size
  assert jax.device_count() % group_size == 0
  return [
      list(j
           for j in range(i, i + group_size))
      for i in range(0, jax.device_count(), group_size)
  ]
