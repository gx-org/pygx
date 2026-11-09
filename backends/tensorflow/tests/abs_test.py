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

"""Test GX math.Abs with a TensorFlow backend.

Ported from UnaryOpTest.testFloatBasic and UnaryOpTest.testFloatEmpty in
third_party/tensorflow/python/kernel_tests/math_ops/cwise_ops_unary_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfAbsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _compare(self, x: np.ndarray):
    x_tf = tf.constant(x)
    expected = np.abs(x)

    result = gxtftests.AbsFloat32(tf.shape(x_tf), x_tf)

    self.assertShapeEqual(expected, result)
    self.assertAllClose(result, expected)

  def test_abs_float32(self):
    self._compare(np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32))

  def test_abs_empty_float32(self):
    self._compare(np.empty((2, 0, 5), dtype=np.float32))

  def test_tf_abs_in_tf_function(self):
    """Tests AbsFloat32 inside tf.function and XLA."""

    def fn(t):
      return gxtftests.AbsFloat32(tf.shape(t), t)

    x = tf.constant([-2.0, 0.0, 3.0])
    expected = tf.math.abs(x)
    self.assertAllClose(tf.function(fn)(x), expected)
    self.assertAllClose(tf.function(fn, jit_compile=True)(x), expected)

  def test_tf_abs_gradient(self):
    """Tests gradients through AbsFloat32 in eager, graph, and XLA modes."""

    def grad_fn(t):
      with tf.GradientTape() as tape:
        tape.watch(t)
        y = gxtftests.AbsFloat32(tf.shape(t), t)
      return tape.gradient(y, t)

    x = tf.constant([-2.0, 3.0])
    self.assertAllClose(grad_fn(x), [-1.0, 1.0])
    self.assertAllClose(tf.function(grad_fn)(x), [-1.0, 1.0])
    self.assertAllClose(tf.function(grad_fn, jit_compile=True)(x), [-1.0, 1.0])


if __name__ == "__main__":
  absltest.main()
