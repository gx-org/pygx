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

"""Test GX constants with a TensorFlow backend.

Ported from ConstantTest (testFloat, testDouble, testInt32, testInt64) in
third_party/tensorflow/python/kernel_tests/array_ops/constant_op_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


def _normal() -> np.ndarray:
  return (100 * np.random.normal(size=30)).reshape([2, 3, 5])


def _arange() -> np.ndarray:
  return np.arange(-15, 15).reshape([2, 3, 5])


class TfConstantTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _test_scalar(self, gx_func, x: np.ndarray, expected: np.ndarray):
    x_tf = tf.constant(x)

    result = gx_func(tf.shape(x_tf), x_tf)

    self.assertEqual(result.dtype, x_tf.dtype)
    self.assertAllEqual(result, expected)
    fn = lambda a: gx_func(tf.shape(a), a)
    self.assertAllEqual(tf.function(fn)(x_tf), expected)
    self.assertAllEqual(tf.function(fn, jit_compile=True)(x_tf), expected)

  def test_float64(self):
    x = _normal().astype(np.float64)
    self._test_scalar(gxtftests.AddConstFloat64, x, x + np.float64(0.5))

  def test_int32(self):
    x = _normal().astype(np.int32)
    self._test_scalar(gxtftests.AddConstInt32, x, x + np.int32(7))

  def test_int64(self):
    x = _normal().astype(np.int64)
    self._test_scalar(gxtftests.AddConstInt64, x, x + np.int64(7))

  def test_uint32(self):
    x = np.abs(_normal()).astype(np.uint32)
    self._test_scalar(gxtftests.AddConstUint32, x, x + np.uint32(7))

  def test_uint64(self):
    x = np.abs(_normal()).astype(np.uint64)
    self._test_scalar(gxtftests.AddConstUint64, x, x + np.uint64(7))

  def test_array_float32(self):
    x = _normal().astype(np.float32)

    result = gxtftests.AddConstArrayFloat32(tf.constant(x))

    self.assertEqual(result.dtype, tf.float32)
    self.assertAllClose(result, _arange().astype(np.float32) + x)

  def test_array_int64(self):
    x = _normal().astype(np.int64)

    result = gxtftests.AddConstArrayInt64(tf.constant(x))

    self.assertEqual(result.dtype, tf.int64)
    self.assertAllEqual(result, _arange().astype(np.int64) + x)


if __name__ == "__main__":
  absltest.main()
