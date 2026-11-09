import numpy as np

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import dyngx
from google3.third_party.gxlang.pygx.testing import pygxtesting

MINFLOAT32 = 1.401298464324817070923729583289916131280e-45
MAXFLOAT32 = 3.40282346638528859811704183484516925440e38

MINFLOAT64 = 4.9406564584124654417656879286822137236505980e-324
MAXFLOAT64 = 1.79769313486231570814527423731704356798070e308

MININT32 = -2147483648 + 1
MAXINT32 = 2147483647

MININT64 = -9223372036854775808 + 1
MAXINT64 = 9223372036854775807

MAXUINT32 = 4294967295
MAXUINT64 = 18446744073709551615


class GXDTypesTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._runtime = pygxtesting.new_graph_runtime()
    self._device = self._runtime.device(0)
    self._package = dyngx.build_package(
        self._device,
        "github.com/gx-org/gx/tests/bindings/dtypes",
    )

  def test_bool(self):
    value = self._package.Bool(True)
    self.assertFalse(value.to_host())

  def test_float32(self):
    value = self._package.Float32(MAXFLOAT32)
    self.assertEqual(value.to_host(), -MAXFLOAT32)

  def test_float64(self):
    value = self._package.Float64(MAXFLOAT64)
    self.assertEqual(value.to_host(), -MAXFLOAT64)

  def test_int(self):
    value = self._package.Int(MAXINT64)
    self.assertEqual(value.to_host(), -MAXINT64)

  def test_int32(self):
    value = self._package.Int32(MAXINT32)
    self.assertEqual(value.to_host(), -MAXINT32)

  def test_int64(self):
    value = self._package.Int64(MAXINT64)
    self.assertEqual(value.to_host(), -MAXINT64)

  def test_uint32(self):
    value = self._package.Uint32(MAXUINT32)
    self.assertEqual(value.to_host(), 1)

  def test_uint64(self):
    value = self._package.Uint64(MAXUINT64)
    self.assertEqual(value.to_host(), 1)

  def test_array_bool(self):
    want = np.array([True, False, True], dtype=np.bool)
    value = self._package.ArrayBool(np.logical_not(want))
    np.testing.assert_array_equal(value.to_host(), want)

  def test_array_float32(self):
    want = np.array(
        [[-MAXFLOAT32, MAXFLOAT32], [MINFLOAT32, -MINFLOAT32]], dtype=np.float32
    )
    value = self._package.ArrayFloat32(-want)
    np.testing.assert_array_equal(value.to_host(), want)

  def test_array_float64(self):
    want = np.array(
        [[-MAXFLOAT64, MAXFLOAT64], [MINFLOAT64, -MINFLOAT64]], dtype=np.float64
    )
    value = self._package.ArrayFloat64(-want)
    np.testing.assert_array_equal(value.to_host(), want)

  def test_array_int(self):
    want = np.array(
        [[-MAXINT64, MAXINT64], [MININT64, -MININT64]], dtype=np.int64
    )
    value = self._package.ArrayInt(-want)
    np.testing.assert_array_equal(value.to_host(), want)

  def test_array_int32(self):
    want = np.array(
        [[-MAXINT32, MAXINT32], [MININT32, -MININT32]], dtype=np.int32
    )
    value = self._package.ArrayInt32(-want)
    np.testing.assert_array_equal(value.to_host(), want)

  def test_array_int64(self):
    want = np.array(
        [[-MAXINT64, MAXINT64], [MININT64, -MININT64]], dtype=np.int64
    )
    value = self._package.ArrayInt64(-want)
    np.testing.assert_array_equal(value.to_host(), want)

  def test_array_uint32(self):
    want = np.array([1, MAXUINT32], dtype=np.uint32)
    value = self._package.ArrayUint32(-want)
    np.testing.assert_array_equal(value.to_host(), want)

  def test_array_uint64(self):
    want = np.array([1, MAXUINT64], dtype=np.uint64)
    value = self._package.ArrayUint64(-want)
    np.testing.assert_array_equal(value.to_host(), want)


if __name__ == "__main__":
  googletest.main()
