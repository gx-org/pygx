"""Test GX num.Iota with a TensorFlow backend.

Ported from RangeTest (testLimitOnly, testEmpty) in
third_party/tensorflow/python/kernel_tests/array_ops/init_ops_test.py.
GX iota generalizes tf.range to arrays with several axes.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfIotaTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _eval(self, gx_func):
    res = gx_func()
    self.assertAllEqual(res, tf.function(gx_func)())
    self.assertAllEqual(res, tf.function(gx_func, jit_compile=True)())
    return res

  def test_limit_only(self):
    result = self._eval(gxtftests.Iota5)

    self.assertEqual(result.dtype, tf.int64)
    self.assertAllEqual(result, np.arange(5))

  def test_empty(self):
    result = self._eval(gxtftests.Iota0)

    self.assertEqual(result.dtype, tf.int64)
    self.assertAllEqual(result.shape, [0])

  def test_axis0(self):
    result = self._eval(gxtftests.IotaAxis0)

    self.assertEqual(result.dtype, tf.int64)
    self.assertAllEqual(
        result, np.broadcast_to(np.arange(4).reshape(4, 1), [4, 3])
    )

  def test_axis1(self):
    result = self._eval(gxtftests.IotaAxis1)

    self.assertEqual(result.dtype, tf.int64)
    self.assertAllEqual(
        result, np.broadcast_to(np.arange(3).reshape(1, 3), [4, 3])
    )


if __name__ == "__main__":
  absltest.main()
