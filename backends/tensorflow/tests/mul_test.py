"""Test GX multiplication with a TensorFlow backend.

Ported from BinaryOpTest.testFloatBasic in
third_party/tensorflow/python/kernel_tests/math_ops/cwise_ops_binary_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfMulTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)
    self.x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
    self.y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)

  def test_mul_float32(self):
    x = tf.constant(self.x)
    y = tf.constant(self.y)
    expected = np.multiply(self.x, self.y)

    result = gxtftests.MulFloat32(tf.shape(x), x, y)

    self.assertShapeEqual(expected, result)
    self.assertAllClose(result, expected)
    fn = lambda a, b: gxtftests.MulFloat32(tf.shape(a), a, b)
    for f in (tf.function(fn), tf.function(fn, jit_compile=True)):
      with tf.GradientTape() as tape:
        tape.watch([x, y])
        res = f(x, y)
      self.assertAllClose(res, expected)
      self.assertAllClose(tape.gradient(res, [x, y]), [self.y, self.x])

  def test_mul_by_constant_float32(self):
    x = tf.constant(self.x)
    expected = np.multiply(np.float32(2), self.x)

    result = gxtftests.MulTwoFloat32(tf.shape(x), x)

    self.assertShapeEqual(expected, result)
    self.assertAllClose(result, expected)
    fn = lambda a: gxtftests.MulTwoFloat32(tf.shape(a), a)
    self.assertAllClose(tf.function(fn)(x), expected)
    self.assertAllClose(tf.function(fn, jit_compile=True)(x), expected)


if __name__ == "__main__":
  absltest.main()
