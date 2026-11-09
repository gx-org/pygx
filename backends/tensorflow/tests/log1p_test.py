"""Test GX math.Log1p with a TensorFlow backend.

Ported from UnaryOpTest.testFloatBasic and UnaryOpTest.testFloatEmpty in
third_party/tensorflow/python/kernel_tests/math_ops/cwise_ops_unary_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfLog1pTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _compare(self, x: np.ndarray):
    x_tf = tf.constant(x)
    expected = np.log1p(x)

    result = gxtftests.Log1pFloat32(tf.shape(x_tf), x_tf)

    self.assertShapeEqual(expected, result)
    self.assertAllClose(result, expected)
    fn = lambda a: gxtftests.Log1pFloat32(tf.shape(a), a)
    for f in (tf.function(fn), tf.function(fn, jit_compile=True)):
      with tf.GradientTape() as tape:
        tape.watch(x_tf)
        res = f(x_tf)
      self.assertAllClose(res, expected)
      if x.size:
        self.assertAllClose(tape.gradient(res, x_tf), 1.0 / (1.0 + x))

  def test_log1p_float32(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    z = (x + 15.5).astype(np.float32)  # all positive
    self._compare(z)

  def test_log1p_empty_float32(self):
    self._compare(np.empty((2, 0, 5), dtype=np.float32))


if __name__ == "__main__":
  absltest.main()
