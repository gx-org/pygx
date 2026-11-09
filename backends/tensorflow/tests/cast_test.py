"""Test GX data type conversions with a TensorFlow backend.

Ported from CastOpTest.testBasic in
third_party/tensorflow/python/kernel_tests/array_ops/cast_op_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfCastTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _test(self, x: np.ndarray, dtype: type[np.generic], gx_func):
    expected = x.astype(dtype)

    result = gx_func(tf.constant(x))

    self.assertEqual(result.dtype, tf.as_dtype(dtype))
    self.assertAllEqual(result, expected)
    self.assertAllEqual(tf.function(gx_func)(tf.constant(x)), expected)
    self.assertAllEqual(
        tf.function(gx_func, jit_compile=True)(tf.constant(x)), expected
    )

  def test_float32_to_float64(self):
    x = np.arange(-10, 10).reshape(2, 10).astype(np.float32)
    self._test(x, np.float64, gxtftests.CastFloat32ToFloat64)

  def test_float32_to_int32(self):
    x = np.arange(-10, 10).reshape(2, 10).astype(np.float32)
    self._test(x, np.int32, gxtftests.CastFloat32ToInt32)

  def test_float32_to_int64(self):
    x = np.linspace(-10, 10, 17).astype(np.float32)
    self._test(x, np.int64, gxtftests.CastFloat32ToInt64)

  def test_bool_to_float32(self):
    x = np.arange(-10, 10).reshape(2, 10).astype(np.float32)
    expected = x.astype(np.bool_).astype(np.float32)

    result = gxtftests.CastBoolToFloat32(tf.constant(x))

    self.assertEqual(result.dtype, tf.float32)
    self.assertAllEqual(result, expected)


if __name__ == "__main__":
  absltest.main()
