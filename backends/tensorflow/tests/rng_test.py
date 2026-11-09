"""Test GX Philox random bit generation with a TensorFlow backend.

Ported from StatelessOpsTest (testDeterminismInt, testGetKeyCounterAlg) in
third_party/tensorflow/python/kernel_tests/random/stateless_random_ops_test.py.

The GX Philox state is 3 uint64: a key (1 uint64) followed by a counter
(2 uint64), matching the key and counter of TensorFlow stateless random ops.
"""

import itertools

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


def _state(key: int, counter: int) -> tf.Tensor:
  return tf.constant([key, counter, 0], dtype=tf.uint64)


class TfRngTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def test_shape_and_dtype(self):
    a32, b32 = gxtftests.RandUint32(_state(7, 17))
    a64, b64 = gxtftests.RandUint64(_state(7, 17))

    self.assertEqual(a32.dtype, tf.uint32)
    self.assertEqual(b32.dtype, tf.uint32)
    self.assertAllEqual(a32.shape, [2, 5])
    self.assertAllEqual(b32.shape, [2, 5])
    self.assertEqual(a64.dtype, tf.uint64)
    self.assertEqual(b64.dtype, tf.uint64)
    self.assertAllEqual(a64.shape, [3])
    self.assertAllEqual(b64.shape, [3])

  def test_state_is_updated(self):
    # Consecutive draws from the same generator must differ.
    a32, b32 = gxtftests.RandUint32(_state(7, 17))
    a64, b64 = gxtftests.RandUint64(_state(7, 17))

    self.assertFalse(np.all(a32.numpy() == b32.numpy()))
    self.assertFalse(np.all(a64.numpy() == b64.numpy()))
    for jit in (False, True):
      self.assertAllEqual(
          tf.function(gxtftests.RandUint32, jit_compile=jit)(_state(7, 17)),
          (a32, b32),
      )
      self.assertAllEqual(
          tf.function(gxtftests.RandUint64, jit_compile=jit)(_state(7, 17)),
          (a64, b64),
      )

  def test_determinism(self):
    # Values should be equal iff the seeds are equal.
    seeds = list(itertools.product(range(3), range(3))) * 2
    values = [
        (seed, gxtftests.RandUint32(_state(*seed))[0].numpy()) for seed in seeds
    ]
    for s0, v0 in values:
      for s1, v1 in values:
        self.assertEqual(s0 == s1, np.all(v0 == v1))

  def test_source(self):
    # A generator seeded by a constant always draws the same values.
    a = gxtftests.RandUint32FromSource()
    b = gxtftests.RandUint32FromSource()

    self.assertEqual(a.dtype, tf.uint32)
    self.assertAllEqual(a.shape, [2, 5])
    self.assertAllEqual(a, b)


if __name__ == "__main__":
  absltest.main()
