# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test GX ID function with a TensorFlow backend."""

from absl.testing import absltest
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfIdTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

  def tf_id(self, a):
    """Takes an argument and returns it using TensorFlow."""
    tensor_a = tf.convert_to_tensor(a)
    return gxtftests.ID(tensor_a)

  def test_tf_id_with_scalars(self):
    """Tests ID with a simple scalar float value."""
    x = 5.0
    expected = 5.0

    result = self.tf_id(x)

    self.assertAllClose(result, expected)

  def test_tf_id_with_tensors(self):
    """Tests ID directly using TensorFlow Tensors."""
    x = tf.constant([1.0, 2.0])
    expected = [1.0, 2.0]

    result = self.tf_id(x)

    self.assertAllClose(result, expected)

  def test_tf_id_float32_with_tensors(self):
    """Tests IDFloat32 with TensorFlow Tensors."""
    x = tf.constant([1.0, 2.0])
    expected = [1.0, 2.0]

    result = gxtftests.IDFloat32(tf.shape(x), x)

    self.assertAllClose(result, expected)

  def test_tf_id_in_tf_function(self):
    """Tests ID and IDFloat32 inside tf.function and XLA."""

    def fn(t):
      return gxtftests.ID(t), gxtftests.IDFloat32(tf.shape(t), t)

    x = tf.constant([1.0, 2.0])
    for compiled_fn in (tf.function(fn), tf.function(fn, jit_compile=True)):
      res_id, res_id_f32 = compiled_fn(x)
      self.assertAllClose(res_id, x)
      self.assertAllClose(res_id_f32, x)

  def test_tf_id_gradient(self):
    """Tests gradients through IDFloat32 in eager, graph, and XLA modes."""

    def grad_fn(t):
      with tf.GradientTape() as tape:
        tape.watch(t)
        y = gxtftests.IDFloat32(tf.shape(t), t)
      return tape.gradient(y, t)

    x = tf.constant([1.0, 2.0])
    self.assertAllClose(grad_fn(x), [1.0, 1.0])
    self.assertAllClose(tf.function(grad_fn)(x), [1.0, 1.0])
    self.assertAllClose(tf.function(grad_fn, jit_compile=True)(x), [1.0, 1.0])


if __name__ == "__main__":
  absltest.main()
