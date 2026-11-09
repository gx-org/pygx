"""Test GX axlengths builtin with a TensorFlow backend."""

from absl.testing import absltest
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfAxLengthsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def test_lens(self):
    """Tests Lens with a 3D TensorFlow Tensor."""
    x = tf.zeros((3, 4, 5), dtype=tf.float32)

    result = gxtftests.Lens(tf.shape(x), x)

    self.assertAllEqual(result, (3, 4, 5))

  def test_lens_in_tf_function(self):
    """Tests Lens inside tf.function and XLA."""

    def fn(t):
      return gxtftests.Lens(tf.shape(t), t)

    x = tf.zeros((3, 4, 5), dtype=tf.float32)
    for compiled_fn in (tf.function(fn), tf.function(fn, jit_compile=True)):
      self.assertAllEqual(compiled_fn(x), (3, 4, 5))

  def test_lens_unknown_axis_length(self):
    """Tests Lens with an unknown (None) axis length inside tf.function."""

    def fn(t):
      return gxtftests.Lens(t.shape.as_list(), t)

    sig = [tf.TensorSpec(shape=[None, 4, 5], dtype=tf.float32)]
    x = tf.zeros((3, 4, 5), dtype=tf.float32)
    for compiled_fn in (
        tf.function(fn, input_signature=sig),
        tf.function(fn, input_signature=sig, jit_compile=True),
    ):
      self.assertAllEqual(compiled_fn(x), (-1, 4, 5))


if __name__ == "__main__":
  absltest.main()
