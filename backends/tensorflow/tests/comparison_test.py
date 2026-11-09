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

"""Test GX comparison operators with a TensorFlow backend.

Ported from ComparisonOpTest (testScalarCompareScalar,
testTensorCompareTensor) in
third_party/tensorflow/python/kernel_tests/math_ops/cwise_ops_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfComparisonTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _compare(self, x: np.ndarray, y: np.ndarray, np_func, gx_func):
    x_tf = tf.constant(x)
    y_tf = tf.constant(y)
    expected = np_func(x, y)

    result = gx_func(tf.shape(x_tf), x_tf, y_tf)

    self.assertEqual(result.dtype, tf.bool)
    self.assertAllEqual(result, expected)
    fn = lambda a, b: gx_func(tf.shape(a), a, b)
    self.assertAllEqual(tf.function(fn)(x_tf, y_tf), expected)
    self.assertAllEqual(tf.function(fn, jit_compile=True)(x_tf, y_tf), expected)

  def test_scalar_compare_scalar(self):
    data = [-1, 0, 1]
    for x in data:
      for y in data:
        with self.subTest(x=x, y=y):
          xt = np.array([x], dtype=np.float32)
          yt = np.array([y], dtype=np.float32)
          self._compare(xt, yt, np.less, gxtftests.LessFloat32)
          self._compare(xt, yt, np.less_equal, gxtftests.LessEqualFloat32)
          self._compare(xt, yt, np.greater, gxtftests.GreaterFloat32)
          self._compare(xt, yt, np.greater_equal, gxtftests.GreaterEqualFloat32)
          self._compare(xt, yt, np.equal, gxtftests.EqualFloat32)
          self._compare(xt, yt, np.not_equal, gxtftests.NotEqualFloat32)

  def test_tensor_compare_tensor(self):
    x = np.linspace(-15, 15, 6).reshape(1, 3, 2).astype(np.float32)
    y = np.linspace(20, -10, 6).reshape(1, 3, 2).astype(np.float32)
    self._compare(x, y, np.less, gxtftests.LessFloat32)
    self._compare(x, y, np.less_equal, gxtftests.LessEqualFloat32)
    self._compare(x, y, np.greater, gxtftests.GreaterFloat32)
    self._compare(x, y, np.greater_equal, gxtftests.GreaterEqualFloat32)
    self._compare(x, y, np.equal, gxtftests.EqualFloat32)
    self._compare(x, y, np.not_equal, gxtftests.NotEqualFloat32)

  def test_scalar_compare_tensor(self):
    p = np.float32(0.25)
    x = np.linspace(0, 1, 6).reshape(2, 3).astype(np.float32)
    x_tf = tf.constant(x)

    result = gxtftests.GreaterEqualScalarFloat32(
        tf.shape(x_tf), tf.constant(p), x_tf
    )

    self.assertAllEqual(result, np.greater_equal(p, x))


if __name__ == "__main__":
  absltest.main()
