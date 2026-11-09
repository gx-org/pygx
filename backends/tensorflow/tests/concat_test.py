"""Test GX shape.Concat with a TensorFlow backend.

Ported from ConcatOpTest (testHStack, testVStack, test4DStack, testInt32GPU)
in third_party/tensorflow/python/kernel_tests/array_ops/concat_op_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfConcatTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _eval(self, gx_func, p1, p2):
    a, b = tf.constant(p1), tf.constant(p2)
    res = gx_func(a, b)
    self.assertAllEqual(res, tf.function(gx_func)(a, b))
    self.assertAllEqual(res, tf.function(gx_func, jit_compile=True)(a, b))
    return res

  def test_hstack(self):
    p1 = np.random.rand(4, 4).astype(np.float32)
    p2 = np.random.rand(4, 4).astype(np.float32)

    result = self._eval(gxtftests.ConcatAxis0, p1, p2)

    self.assertAllEqual(result.shape, [8, 4])
    self.assertAllEqual(result[:4, :], p1)
    self.assertAllEqual(result[4:, :], p2)

  def test_vstack(self):
    p1 = np.random.rand(4, 4).astype(np.float32)
    p2 = np.random.rand(4, 4).astype(np.float32)

    result = self._eval(gxtftests.ConcatAxis1, p1, p2)

    self.assertAllEqual(result.shape, [4, 8])
    self.assertAllEqual(result[:, :4], p1)
    self.assertAllEqual(result[:, 4:], p2)

  def test_4d_stack(self):
    p1 = np.random.rand(2, 3, 1, 1).astype(np.float32)
    p2 = np.random.rand(2, 3, 4, 1).astype(np.float32)

    result = self._eval(gxtftests.Concat4D, p1, p2)

    self.assertAllEqual(result.shape, [2, 3, 5, 1])
    self.assertAllEqual(result[:, :, :1, :], p1)
    self.assertAllEqual(result[:, :, 1:, :], p2)

  def test_int32(self):
    p1 = (np.random.rand(2, 3) * 10).astype(np.int32)
    p2 = (np.random.rand(2, 3) * 10).astype(np.int32)

    result = self._eval(gxtftests.ConcatInt32, p1, p2)

    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)


if __name__ == "__main__":
  absltest.main()
