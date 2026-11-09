"""Test GX shape.Broadcast with a TensorFlow backend.

Ported from BroadcastToTest (testBroadcastToBasic,
testBroadcastToShapeInnerDim) in
third_party/tensorflow/python/kernel_tests/array_ops/broadcast_to_ops_test.py.

GX broadcasting requires the same number of axes for the source and the
target arrays: [1, 2, 3] is passed as [[1, 2, 3]] (same numpy result).
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfBroadcastTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _eval(self, gx_func, x):
    res = gx_func(x)
    self.assertAllEqual(res, tf.function(gx_func)(x))
    self.assertAllEqual(res, tf.function(gx_func, jit_compile=True)(x))
    return res

  def test_broadcast_to_basic(self):
    for dtype, gx_func in [
        (np.float32, gxtftests.BroadcastFloat32),
        (np.int32, gxtftests.BroadcastInt32),
    ]:
      with self.subTest(dtype=dtype):
        x = np.array([[1, 2, 3]], dtype=dtype)

        v_tf = self._eval(gx_func, tf.constant(x))

        self.assertAllEqual(v_tf, np.broadcast_to(x, [3, 3]))

  def test_broadcast_to_shape_inner_dim(self):
    input_shape = [2, 1, 3]
    output_shape = [2, 5, 3]
    x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)

    v_tf = self._eval(gxtftests.BroadcastInnerAxisInt32, tf.constant(x))

    self.assertAllEqual(v_tf, np.broadcast_to(x, output_shape))


if __name__ == "__main__":
  absltest.main()
