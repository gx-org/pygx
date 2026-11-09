"""Test GX shape.Gather with a TensorFlow backend.

Ported from GatherNdTest (testSimpleDtype,
testParamsRankLargerThanIndexSlices, testHigherRankParams) in
third_party/tensorflow/python/kernel_tests/array_ops/gather_nd_op_test.py.
"""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfGatherTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def _eval(self, gx_func, params, indices):
    res = gx_func(params, indices)
    self.assertAllEqual(res, tf.function(gx_func)(params, indices))
    self.assertAllEqual(
        res, tf.function(gx_func, jit_compile=True)(params, indices)
    )
    return res

  def test_simple(self):
    params = tf.constant(np.array([8, 1, 2, 3, 7, 5], dtype=np.float32))
    indices = tf.constant([[4], [4], [0]], dtype=tf.int64)

    gather_nd_t = self._eval(gxtftests.GatherSimple, params, indices)

    self.assertAllEqual(np.array([7, 7, 8], dtype=np.float32), gather_nd_t)
    self.assertEqual([3], gather_nd_t.get_shape())

  def test_params_rank_larger_than_index_slices(self):
    params = np.array(
        [[-8, -1, -2, -3, -7, -5], [8, 1, 2, 3, 7, 5]], dtype=np.float32
    ).T
    indices = tf.constant([[4], [4], [0]], dtype=tf.int64)

    gather_nd_t = self._eval(
        gxtftests.GatherSlices, tf.constant(params), indices
    )

    self.assertEqual([3, 2], gather_nd_t.get_shape())
    self.assertAllEqual(np.array([[-7, 7], [-7, 7], [-8, 8]]), gather_nd_t)

  def test_higher_rank_params(self):
    shape = (10, 20, 5, 1, 17)
    params = np.random.rand(*shape).astype(np.float32)
    indices = np.vstack(
        [np.random.randint(0, s, size=2000) for s in shape]
    ).T.astype(np.int64)

    gather_nd_t = self._eval(
        gxtftests.GatherHigherRank, tf.constant(params), tf.constant(indices)
    )

    expected = params[tuple(indices.T)]
    self.assertAllEqual(expected, gather_nd_t)
    self.assertEqual([2000], gather_nd_t.get_shape())


if __name__ == "__main__":
  absltest.main()
