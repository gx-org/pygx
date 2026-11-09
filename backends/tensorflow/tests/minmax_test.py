"""Test GX math.Max and math.Min with a TensorFlow backend.

Ported from MinMaxOpTest (testBasic, testNaNPropagation, testScalar) in
third_party/tensorflow/python/kernel_tests/math_ops/cwise_ops_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfMinMaxTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _eval(self, gx_func, *args):
    res = gx_func(*args)
    self.assertAllEqual(res, tf.function(gx_func)(*args))
    self.assertAllEqual(res, tf.function(gx_func, jit_compile=True)(*args))
    return res

  def test_basic(self):
    x = (np.random.rand(1, 3, 2) * 100.0).astype(np.float32)
    y = (np.random.rand(1, 3, 2) * 100.0).astype(np.float32)

    tf_min = self._eval(gxtftests.MinFloat32, tf.constant(x), tf.constant(y))
    tf_max = self._eval(gxtftests.MaxFloat32, tf.constant(x), tf.constant(y))

    self.assertAllEqual(tf_min, np.minimum(x, y))
    self.assertAllEqual(tf_max, np.maximum(x, y))

  def test_nan_propagation(self):
    x = np.array([1.0, np.nan, 1.0, np.nan], dtype=np.float32)
    y = np.array([1.0, 1.0, np.nan, np.nan], dtype=np.float32)

    tf_min = tf.function(gxtftests.MinNaNFloat32)(
        tf.constant(x), tf.constant(y)
    )
    tf_max = tf.function(gxtftests.MaxNaNFloat32)(
        tf.constant(x), tf.constant(y)
    )

    self.assertAllEqual(tf_min, np.minimum(x, y))
    self.assertAllEqual(tf_max, np.maximum(x, y))

  def test_scalar(self):
    x = (np.random.rand(1, 3, 2) * 100.0).astype(np.float32)
    y = np.float32(np.random.rand(1).item() * 100.0)

    tf_min = self._eval(
        gxtftests.MinScalarFloat32, tf.constant(x), tf.constant(y)
    )
    tf_max = self._eval(
        gxtftests.MaxScalarFloat32, tf.constant(x), tf.constant(y)
    )

    self.assertAllEqual(tf_min, np.minimum(x, y))
    self.assertAllEqual(tf_max, np.maximum(x, y))

  def test_clip(self):
    for p in [-0.5, 0.0, 0.3, 1.0, 1.5]:
      with self.subTest(p=p):
        result = self._eval(
            gxtftests.ClipFloat32, tf.constant(p, dtype=tf.float32)
        )
        self.assertAllClose(result, np.clip(np.float32(p), 0.0, 1.0))


if __name__ == "__main__":
  absltest.main()
