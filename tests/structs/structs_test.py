"""Test GX dynamic bindings."""

import numpy as np

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import dyngx
from google3.third_party.gxlang.pygx.testing import pygxtesting


class PyStructsTest(googletest.TestCase):

  PKG_PATH = "github.com/gx-org/pygx/tests/structs"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._runtime = pygxtesting.new_graph_runtime()
    self._dev = self._runtime.device(0)
    self._structs = dyngx.build_package(self._dev, self.PKG_PATH)
    self.assertIsNotNone(self._structs)

  def testDir(self):
    sa = self._structs.New(5)
    self.assertIsInstance(sa, dyngx.Instance)
    self.assertEqual(dir(sa), ["A", "B"])
    self.assertIsInstance(sa.B, dyngx.Instance)
    self.assertEqual(dir(sa.B), ["B"])

  def testFields(self):
    sa = self._structs.New(np.int32(5))
    self.assertIsInstance(sa, dyngx.Instance)
    self.assertIsInstance(sa.B, dyngx.Instance)
    np.testing.assert_array_equal(
        sa.A.to_host(), np.array([6, 7], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        sa.B.B.to_host(), np.array([8, 9, 10], dtype=np.float32)
    )

  def testMultipleResults(self):
    scalar, sa, arr = self._structs.NewWithArrays(
        np.int32(5), np.array([10, 20], dtype=np.int32)
    )
    self.assertEqual(scalar.to_host(), 5)
    self.assertIsInstance(sa, dyngx.Instance)
    self.assertEqual(dir(sa), ["A", "B"])
    self.assertIsInstance(sa.B, dyngx.Instance)
    np.testing.assert_array_equal(
        sa.A.to_host(), np.array([6, 7], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        sa.B.B.to_host(), np.array([8, 9, 10], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        arr.to_host(), np.array([15, 25], dtype=np.int32)
    )

  def testSetValue(self):
    sa = self._structs.New(5)
    self.assertEqual(
        str(sa),
        """structs.SA{
	A: DeviceArray{DeviceID: 0}: [2]int32{6, 7},
	B: structs.SB{
		B: DeviceArray{DeviceID: 0}: [3]float32{8, 9, 10},
	},
}""",
    )
    self.assertEqual(
        str(sa.B),
        """structs.SB{
	B: DeviceArray{DeviceID: 0}: [3]float32{8, 9, 10},
}""",
    )


if __name__ == "__main__":
  googletest.main()
