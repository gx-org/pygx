"""Test GX dynamic bindings."""

import numpy as np

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import dyngx
from google3.third_party.gxlang.pygx.testing import pygxtesting


class PygxTest(googletest.TestCase):
  BASIC_PACKAGE = "github.com/gx-org/gx/tests/bindings/basic"
  CARTPOLE_PACKAGE = (
      "github.com/gx-org/gx/tests/bindings/cartpole"
  )
  GENERIC_PACKAGE = (
      "github.com/gx-org/gx/tests/bindings/generics"
  )
  PARAMETERS_PACKAGE = (
      "github.com/gx-org/gx/tests/bindings/parameters"
  )
  PKGVARS_PACKAGE = (
      "github.com/gx-org/gx/tests/bindings/pkgvars"
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._runtime = pygxtesting.new_graph_runtime()
    self._dev = self._runtime.device(0)
    self._basic = dyngx.build_package(self._dev, self.BASIC_PACKAGE)
    self._generic = dyngx.build_package(self._dev, self.GENERIC_PACKAGE)
    self._params = dyngx.build_package(self._dev, self.PARAMETERS_PACKAGE)
    self._pkgvars = dyngx.build_package(self._dev, self.PKGVARS_PACKAGE)
    self.assertIsNotNone(self._basic)
    self.assertIsNotNone(self._generic)
    self.assertIsNotNone(self._params)
    self.assertIsNotNone(self._pkgvars)

  def testPackage_ListFunctions(self):
    attrs = dir(self._basic)
    wants = [
        "AddPrivate",
        "New",
        "ReturnArrayFloat32",
        "ReturnFloat32",
        "ReturnMultiple",
    ]
    self.assertSequenceEqual(attrs, wants)

  def testPackage_PkgVarsDir(self):
    attrs = dir(self._pkgvars)
    wants = ["NewTwiceSize", "ReturnVar1", "Size", "Var1"]
    self.assertSequenceEqual(attrs, wants)

  def testStaticVar_Set(self):
    self._pkgvars.Var1.set(42)
    self.assertEqual(42, self._pkgvars.ReturnVar1().to_host())

  def testInterface_ListFunctions(self):
    iface = self._basic.New()
    self.assertEqual(
        str(iface),
        """basic.Basic{
	Int: DeviceArray{DeviceID: 0}: int32(42),
	Float: DeviceArray{DeviceID: 0}: float32(4.2),
	Array: DeviceArray{DeviceID: 0}: [2]float32{4.2, 42},
	privateA: DeviceArray{DeviceID: 0}: int32(2),
	privateB: DeviceArray{DeviceID: 0}: int32(4),
	length: DeviceArray{DeviceID: 0}: int(4),
	index: DeviceArray{DeviceID: 0}: int(2),
}""",
    )
    attrs = dir(iface)
    wants = ["Add", "AddPrivate", "Array", "Float", "Int", "SetFloat"]
    self.assertSequenceEqual(attrs, wants)

  def testReturnNumpyArray(self):
    device_array = self._basic.ReturnArrayFloat32()
    array = device_array.to_host()
    np.testing.assert_array_equal(array, np.array([4.2, 42], dtype=np.float32))

  def testReturnFloat32(self):
    device_value = self._basic.ReturnFloat32()
    value = device_value.to_host()
    self.assertAlmostEqual(value, 4.2, places=6)

  def testPassInt64s(self):
    device_value = self._params.AddInt(100, 200)
    self.assertEqual(300, device_value.to_host())

  def testPassFloat32Arrays(self):
    self._params.Size.set(3)
    device_value = self._params.AddFloat32s(
        np.array([1, 2, 3], dtype=np.float32),
        np.array([10, 20, 30], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        device_value.to_host(), np.array([11, 22, 33], dtype=np.float32)
    )
    device_value = self._params.AddFloat32s(device_value, device_value)
    np.testing.assert_array_equal(
        device_value.to_host(), np.array([22, 44, 66], dtype=np.float32)
    )

  def testStatefulCartpoleStructure(self):
    cartpole_pkg = dyngx.build_package(self._dev, self.CARTPOLE_PACKAGE)
    cartpole = cartpole_pkg.New()
    cartpole = cartpole.Reset()
    for _ in range(5):
      cartpole = cartpole.Step(1.0)
    state = cartpole.State()
    np.testing.assert_allclose(
        state.to_host(),
        np.array([0.039032, 0.976364, -0.058719, -1.481532], dtype=np.float32),
        atol=0.00001,
    )

  def testInstances(self):
    a = self._basic.New()
    b = self._basic.New()
    self.assertEqual(4, a.Add(b).to_host())

  def testGeneric(self):
    a = self._generic.VecF32(3)
    np.testing.assert_equal(a.to_host(), np.array([0, 0, 0], dtype=np.float32))

  def testGenericSlice(self):
    a = self._generic.ArrayF32([2, 3])
    np.testing.assert_equal(
        a.to_host(), np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    )


if __name__ == "__main__":
  googletest.main()
