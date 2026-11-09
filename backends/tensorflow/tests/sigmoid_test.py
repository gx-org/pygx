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

"""Test GX Sigmoid function with a TensorFlow backend."""

from absl.testing import absltest
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfSigmoidTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def test_tf_sigmoid_with_scalars(self):
    """Tests Sigmoid with a simple scalar float value."""
    x = tf.constant(0.0)
    expected = 0.5

    result = gxtftests.Sigmoid(tf.shape(x), x)

    self.assertAllClose(result, expected)

  def test_tf_sigmoid_with_tensors(self):
    """Tests Sigmoid with TensorFlow Tensors."""
    x = tf.constant([0.0, 0.0])
    expected = [0.5, 0.5]

    result = gxtftests.Sigmoid(tf.shape(x), x)

    self.assertAllClose(result, expected)

  def test_tf_sigmoid_in_tf_function(self):
    """Tests Sigmoid inside tf.function and XLA."""

    def fn(logit):
      return gxtftests.Sigmoid(tf.shape(logit), logit)

    x = tf.constant([-2.0, 0.0, 2.0])
    expected = tf.math.sigmoid(x)
    self.assertAllClose(tf.function(fn)(x), expected)
    self.assertAllClose(tf.function(fn, jit_compile=True)(x), expected)

  def test_tf_sigmoid_gradient(self):
    """Tests d/dlogit of Sigmoid with tf.GradientTape in eager, graph, and XLA modes."""
    x = tf.constant([-2.0, 0.0, 2.0])
    with tf.GradientTape() as ref_tape:
      ref_tape.watch(x)
      ref_y = tf.math.sigmoid(x)
    expected_grad = ref_tape.gradient(ref_y, x)

    def grad_fn(logit):
      with tf.GradientTape() as tape:
        tape.watch(logit)
        y = gxtftests.Sigmoid(tf.shape(logit), logit)
      return tape.gradient(y, logit)

    self.assertAllClose(grad_fn(x), expected_grad)
    self.assertAllClose(tf.function(grad_fn)(x), expected_grad)
    self.assertAllClose(
        tf.function(grad_fn, jit_compile=True)(x), expected_grad
    )


if __name__ == "__main__":
  absltest.main()
