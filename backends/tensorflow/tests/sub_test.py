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

"""Test GX subtraction with a TensorFlow backend.

Ported from BinaryOpTest.testFloatBasic in
third_party/tensorflow/python/kernel_tests/math_ops/cwise_ops_binary_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfSubTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)
    self.x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
    self.y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)

  def test_sub_float32(self):
    x = tf.constant(self.x)
    y = tf.constant(self.y)
    expected = np.subtract(self.x, self.y)

    result = gxtftests.SubFloat32(tf.shape(x), x, y)

    self.assertShapeEqual(expected, result)
    self.assertAllClose(result, expected)
    fn = lambda a, b: gxtftests.SubFloat32(tf.shape(a), a, b)
    for f in (tf.function(fn), tf.function(fn, jit_compile=True)):
      with tf.GradientTape() as tape:
        tape.watch([x, y])
        res = f(x, y)
      self.assertAllClose(res, expected)
      self.assertAllClose(
          tape.gradient(res, [x, y]),
          [np.ones_like(self.x), -np.ones_like(self.y)],
      )

  def test_sub_from_constant_float32(self):
    x = tf.constant(self.x)
    expected = np.subtract(np.float32(1), self.x)

    result = gxtftests.SubFromOneFloat32(tf.shape(x), x)

    self.assertShapeEqual(expected, result)
    self.assertAllClose(result, expected)
    fn = lambda a: gxtftests.SubFromOneFloat32(tf.shape(a), a)
    self.assertAllClose(tf.function(fn)(x), expected)
    self.assertAllClose(tf.function(fn, jit_compile=True)(x), expected)


if __name__ == "__main__":
  absltest.main()
