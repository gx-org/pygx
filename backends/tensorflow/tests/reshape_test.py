"""Test GX array reshapes with a TensorFlow backend.

Ported from ReshapeTest (testFloatBasic, testFloatReshapeThreeDimensions) in
third_party/tensorflow/python/kernel_tests/array_ops/reshape_op_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfReshapeTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _test_reshape(self, x: np.ndarray, shape: list[int], gx_func):
    expected = x.reshape(shape)

    result = gx_func(tf.constant(x))

    self.assertShapeEqual(expected, result)
    self.assertAllEqual(result, expected)
    self.assertAllEqual(tf.function(gx_func)(tf.constant(x)), expected)
    self.assertAllEqual(
        tf.function(gx_func, jit_compile=True)(tf.constant(x)), expected
    )

  def test_float_basic(self):
    x = np.arange(1.0, 7.0).reshape([1, 6]).astype(np.float32)
    self._test_reshape(x, [2, 3], gxtftests.ReshapeFloat32To2x3)

  def test_float_reshape_three_dimensions(self):
    x = np.arange(1.0, 28.0).reshape([1, 27]).astype(np.float32)
    self._test_reshape(x, [3, 3, 3], gxtftests.ReshapeFloat32To3x3x3)

  def test_float_flatten(self):
    x = np.arange(1.0, 7.0).reshape([2, 3]).astype(np.float32)
    self._test_reshape(x, [6], gxtftests.ReshapeFloat32To6)


if __name__ == "__main__":
  absltest.main()
