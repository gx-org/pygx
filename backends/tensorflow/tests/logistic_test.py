"""Test GX math.Logistic with a TensorFlow backend.

Ported from UnaryOpTest (testFloatBasic, testFloatEmpty, testDoubleBasic) in
third_party/tensorflow/python/kernel_tests/math_ops/cwise_ops_unary_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfLogisticTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _compare(self, x: np.ndarray, gx_func):
    x_tf = tf.constant(x)
    expected = (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(x.dtype)
    expected_grad = expected * (1.0 - expected)

    fn = lambda a: gx_func(tf.shape(a), a)
    for f in (fn, tf.function(fn), tf.function(fn, jit_compile=True)):
      with tf.GradientTape() as tape:
        tape.watch(x_tf)
        res = f(x_tf)
      self.assertEqual(res.dtype, x_tf.dtype)
      self.assertShapeEqual(expected, res)
      self.assertAllClose(res, expected)
      if x.size:
        self.assertAllClose(tape.gradient(res, x_tf), expected_grad)

  def test_logistic_float32(self):
    self._compare(
        np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32),
        gxtftests.LogisticFloat32,
    )
    self._compare(np.array(0.0, dtype=np.float32), gxtftests.LogisticFloat32)
    self._compare(
        np.array([-100.0, 0.0, 100.0], dtype=np.float32),
        gxtftests.LogisticFloat32,
    )

  def test_logistic_empty_float32(self):
    self._compare(
        np.empty((2, 0, 5), dtype=np.float32), gxtftests.LogisticFloat32
    )

  def test_logistic_float64(self):
    self._compare(
        np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64),
        gxtftests.LogisticFloat64,
    )

  def test_logistic_bfloat16(self):
    x_tf = tf.constant([-1.0, 0.0, 1.0], dtype=tf.bfloat16)
    expected = tf.math.sigmoid(x_tf)
    fn = lambda a: gxtftests.LogisticBFloat16(tf.shape(a), a)
    for f in (fn, tf.function(fn), tf.function(fn, jit_compile=True)):
      res = f(x_tf)
      self.assertEqual(res.dtype, tf.bfloat16)
      self.assertAllClose(res, expected)


if __name__ == "__main__":
  absltest.main()
