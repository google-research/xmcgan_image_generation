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

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from xmcgan.utils import pretrained_model_utils

FLAGS = flags.FLAGS


class PretrainedModelUtilsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on pretrained model functionality."""

  @parameterized.parameters((1, 256, "resnet50"), (2, 128, "resnet50"))
  def test_pretrained_model(self, batch_size, image_size, model_name):
    model, state = pretrained_model_utils.get_pretrained_model(
        model_name, checkpoint_path=None)
    images = np.random.uniform(0, 1, (batch_size, image_size, image_size, 3))
    pool, outputs = pretrained_model_utils.get_pretrained_embs(
        state, model, images=images)
    self.assertAllEqual(pool.shape, (batch_size, 7, 7, 2048))
    self.assertAllEqual(outputs.shape, (batch_size, 1000))


if __name__ == "__main__":
  tf.test.main()
