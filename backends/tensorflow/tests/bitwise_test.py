"""Test GX bitwise operators with a TensorFlow backend.

Ported from BitwiseOpTest (testBinaryOps, testShiftsWithPositiveLHS,
testShiftsWithNegativeLHS) in
third_party/tensorflow/python/ops/bitwise_ops_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests

# Left-hand side used by the unsigned GX test functions.
_LHS = [0, 5, 3, 14]


class TfBitwiseTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _eval(self, gx_func, rhs):
    res = gx_func(rhs)
    self.assertAllEqual(res, tf.function(gx_func)(rhs))
    self.assertAllEqual(res, tf.function(gx_func, jit_compile=True)(rhs))
    return res

  def test_bitwise_and(self):
    for dtype, gx_func in [
        (tf.uint32, gxtftests.BitwiseAndUint32),
        (tf.uint64, gxtftests.BitwiseAndUint64),
    ]:
      with self.subTest(dtype=dtype):
        rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

        result = self._eval(gx_func, rhs)

        self.assertEqual(result.dtype, dtype)
        self.assertAllEqual(result, [0, 0, 3, 10])

  def test_bitwise_or(self):
    for dtype, gx_func in [
        (tf.uint32, gxtftests.BitwiseOrUint32),
        (tf.uint64, gxtftests.BitwiseOrUint64),
    ]:
      with self.subTest(dtype=dtype):
        rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

        result = self._eval(gx_func, rhs)

        self.assertEqual(result.dtype, dtype)
        self.assertAllEqual(result, [5, 5, 7, 15])

  def test_bitwise_xor(self):
    for dtype, gx_func in [
        (tf.uint32, gxtftests.BitwiseXorUint32),
        (tf.uint64, gxtftests.BitwiseXorUint64),
    ]:
      with self.subTest(dtype=dtype):
        rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

        result = self._eval(gx_func, rhs)

        self.assertEqual(result.dtype, dtype)
        self.assertAllEqual(result, [5, 5, 4, 5])

  def test_shift_left(self):
    for dtype, gx_func in [
        (np.uint32, gxtftests.ShiftLeftUint32),
        (np.uint64, gxtftests.ShiftLeftUint64),
    ]:
      with self.subTest(dtype=dtype):
        lhs = np.array(_LHS, dtype=dtype)
        rhs = np.array([5, 0, 7, 3], dtype=dtype)

        result = self._eval(gx_func, tf.constant(rhs))

        self.assertEqual(result.dtype, tf.as_dtype(dtype))
        self.assertAllEqual(result, np.left_shift(lhs, rhs))

  def test_shift_right_logical(self):
    for dtype, gx_func in [
        (np.uint32, gxtftests.ShiftRightLogicalUint32),
        (np.uint64, gxtftests.ShiftRightLogicalUint64),
    ]:
      with self.subTest(dtype=dtype):
        lhs = np.array(_LHS, dtype=dtype)
        rhs = np.array([5, 0, 7, 3], dtype=dtype)

        result = self._eval(gx_func, tf.constant(rhs))

        self.assertEqual(result.dtype, tf.as_dtype(dtype))
        self.assertAllEqual(result, np.right_shift(lhs, rhs))

  def test_shift_right_arithmetic(self):
    for dtype, gx_func in [
        (np.int32, gxtftests.ShiftRightArithmeticInt32),
        (np.int64, gxtftests.ShiftRightArithmeticInt64),
    ]:
      with self.subTest(dtype=dtype):
        lhs = np.array([-1, -5, -3, -14], dtype=dtype)
        rhs = np.array([5, 0, 7, 11], dtype=dtype)

        result = self._eval(gx_func, tf.constant(rhs))

        self.assertEqual(result.dtype, tf.as_dtype(dtype))
        self.assertAllEqual(result, np.right_shift(lhs, rhs))


if __name__ == "__main__":
  absltest.main()
