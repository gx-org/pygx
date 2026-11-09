"""Test GX dtype.Reinterpret (bitcast) with a TensorFlow backend.

Ported from BitcastTest (testSameSize, testUnsignedType) in
third_party/tensorflow/python/kernel_tests/array_ops/bitcast_op_test.py.
"""

import sys

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfBitcastTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _test_bitcast(self, x: np.ndarray, datatype: tf.DType, gx_func):
    tf_ans = gx_func(tf.constant(x))
    out = tf_ans.numpy()
    if sys.byteorder == "little":
      buff_after = memoryview(out).tobytes()
      buff_before = memoryview(x).tobytes()
    else:
      buff_after = memoryview(out.byteswap()).tobytes()
      buff_before = memoryview(x.byteswap()).tobytes()
    self.assertEqual(buff_before, buff_after)
    self.assertEqual(tf_ans.get_shape(), x.shape)
    self.assertEqual(tf_ans.dtype, datatype)
    self.assertAllEqual(tf_ans, tf.function(gx_func)(tf.constant(x)))
    self.assertAllEqual(
        tf_ans, tf.function(gx_func, jit_compile=True)(tf.constant(x))
    )

  def test_same_size(self):
    x = np.random.rand(3, 4)
    self._test_bitcast(x, tf.int64, gxtftests.ReinterpretFloat64ToInt64)

  def test_unsigned_type(self):
    x = np.zeros([3, 4], np.int64)
    self._test_bitcast(x, tf.uint64, gxtftests.ReinterpretInt64ToUint64)

  def test_uint32_to_float32(self):
    x = np.array([0, 1, 0x3F800000, 0xFFFFFFFF], dtype=np.uint32)
    self._test_bitcast(x, tf.float32, gxtftests.ReinterpretUint32ToFloat32)

  def test_float32_to_uint32(self):
    x = np.array([0.0, 1.0, -2.5, 3.25], dtype=np.float32)
    self._test_bitcast(x, tf.uint32, gxtftests.ReinterpretFloat32ToUint32)


if __name__ == "__main__":
  absltest.main()
