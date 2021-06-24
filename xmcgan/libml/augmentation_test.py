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

from absl.testing import parameterized
import jax
import tensorflow as tf
from xmcgan.libml import augmentation


class AugmentationTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the augmentations."""

  @parameterized.parameters((1, 2), (3, 4), (2, 0))
  def test_aug_shift(self, batch_size, padding_size):
    """Tests for shift augmentation."""
    img = tf.random.normal((batch_size, 32, 32, 3))
    rng = jax.random.PRNGKey(1337)
    rng = jax.random.fold_in(rng, jax.host_id())
    img_aug = augmentation.augment_shift(img, padding_size, seed=rng)
    expected_out_shape = img.shape
    self.assertAllEqual(img_aug.shape, expected_out_shape)

    # Test for determinism.
    img_aug2 = augmentation.augment_shift(img, padding_size, seed=rng)
    self.assertAllEqual(img_aug, img_aug2)

  @parameterized.parameters((1, "bilinear"), (2, "nearest"))
  def test_augment_zoom_crop(self, batch_size, resize_method):
    """Tests for zoom crop augmentation."""
    img = tf.random.normal((batch_size, 32, 32, 3))
    rng = jax.random.PRNGKey(1337)
    rng = jax.random.fold_in(rng, jax.host_id())
    img_aug = augmentation.augment_zoom_crop(img, resize_method, seed=rng)
    expected_out_shape = img.shape
    self.assertAllEqual(img_aug.shape, expected_out_shape)

    # Test for determinism.
    img_aug2 = augmentation.augment_zoom_crop(img, resize_method, seed=rng)
    self.assertAllEqual(img_aug, img_aug2)

  @parameterized.parameters((1, "shift", True), (4, "shift", False),
                            (1, "zoom_crop", False),
                            (4, "zoom_crop", True))
  def test_augment(self, batch_size, method, random_flip):
    """Tests for the general augmentation function."""

    img = tf.random.normal((batch_size, 32, 32, 3))
    rng = jax.random.PRNGKey(1337)
    rng = jax.random.fold_in(rng, jax.host_id())
    img_aug = augmentation.augment(img, method, random_flip, seed=rng)
    expected_out_shape = img.shape
    self.assertAllEqual(img_aug.shape, expected_out_shape)

    # Test for determinism.
    img_aug2 = augmentation.augment(img, method, random_flip, seed=rng)
    self.assertAllEqual(img_aug, img_aug2)

if __name__ == "__main__":
  tf.test.main()



