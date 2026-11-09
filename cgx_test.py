# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import gc
import importlib
import sys
from unittest import mock

import numpy as np

from google3.testing.pybase import googletest
from google3.third_party.gxlang.pygx import cgx
from google3.third_party.gxlang.pygx.testing import pygxtesting


class _CgxRuntimeNewResult(ctypes.Structure):
  """Mirrors struct cgx_runtime_new_result defined in cgx.h."""

  _fields_ = [
      ("runtime", ctypes.c_uint64),
      ("error", ctypes.c_uint64),
  ]


class _CgxDeviceGetResult(ctypes.Structure):
  """Mirrors struct cgx_device_get_result defined in cgx.go."""

  _fields_ = [
      ("device", ctypes.c_uint64),
      ("error", ctypes.c_uint64),
  ]


# The cgx C API and pygxtestdeps_graph_runtime (exported by
# //third_party/gxlang/pygx/testing/pygxtestdeps) are linked into the binary.
_c_lib = ctypes.CDLL(None)

_c_lib.cgx_release_reference.argtypes = [ctypes.c_uint64]
_c_lib.cgx_release_reference.restype = ctypes.c_size_t

_c_lib.cgx_device_get.argtypes = [ctypes.c_uint64, ctypes.c_int64]
_c_lib.cgx_device_get.restype = _CgxDeviceGetResult

_c_lib.pygxtestdeps_graph_runtime.argtypes = []
_c_lib.pygxtestdeps_graph_runtime.restype = _CgxRuntimeNewResult

_c_lib.pygx_wrap_py_object.argtypes = [ctypes.c_void_p]
_c_lib.pygx_wrap_py_object.restype = ctypes.c_uint64


class CgxTest(googletest.TestCase):
  BASIC_PACKAGE = "github.com/gx-org/gx/tests/bindings/basic"
  PARAMETERS_PACKAGE = (
      "github.com/gx-org/gx/tests/bindings/parameters"
  )
  PKGVARS_PACKAGE = (
      "github.com/gx-org/gx/tests/bindings/pkgvars"
  )
  TRACE_PACKAGE = "github.com/gx-org/pygx/tests/trace"
  UNEXPORTED_PACKAGE = (
      "github.com/gx-org/gx/tests/bindings/unexported"
  )

  def _new_runtime_handle(self) -> int:
    """Returns a new raw cgx_runtime handle obtained with ctypes."""
    result = _c_lib.pygxtestdeps_graph_runtime()
    self.assertEqual(result.error, 0)
    return result.runtime

  def testRuntime_FromHandle(self):
    handle = self._new_runtime_handle()
    runtime = cgx.Runtime(handle)
    self.assertEqual(runtime.handle, handle)
    device = runtime.get_device(0)
    data = np.array([4.2, 42], dtype=np.float32)
    np.testing.assert_array_equal(
        cgx.Value.from_ndarray(device, data).to_ndarray(), data
    )

  def testRuntime_FinalizedWithItsDevices(self):
    gc.collect()
    handle_count = cgx.handle_count()
    runtime = cgx.Runtime(self._new_runtime_handle())
    device = runtime.get_device(0)
    # Device of the runtime which does not keep the runtime alive.
    result = _c_lib.cgx_device_get(runtime.handle, 0)
    self.assertEqual(result.error, 0)
    detached_device = cgx.Device(result.device)
    data = np.array([4.2, 42], dtype=np.float32)

    del runtime

    # The device keeps the runtime alive: values can be sent to the device.
    np.testing.assert_array_equal(
        cgx.Value.from_ndarray(detached_device, data).to_ndarray(), data
    )

    del device

    # The backend has been finalized: values cannot be sent to the device.
    with self.assertRaises(RuntimeError):
      cgx.Value.from_ndarray(detached_device, data)
    del detached_device
    gc.collect()
    self.assertEqual(cgx.handle_count(), handle_count)

  def testRuntime_FromZeroHandle(self):
    with self.assertRaisesRegex(ValueError, "invalid GX runtime handle"):
      cgx.Runtime(0)

  def testRuntime_FromReleasedHandle(self):
    handle = self._new_runtime_handle()
    _c_lib.cgx_release_reference(handle)
    with self.assertRaisesRegex(ValueError, "invalid GX runtime handle"):
      cgx.Runtime(handle)

  def testRuntime_FromShapeHandle(self):
    shape = cgx.Shape.new(cgx.Kind.CGX_FLOAT32, ())
    with self.assertRaisesRegex(ValueError, "invalid GX runtime handle"):
      cgx.Runtime(shape.handle)

  def testGetDevice(self):
    runtime = pygxtesting.new_graph_runtime()
    dev = runtime.get_device(0)
    self.assertNotEqual(dev.handle, 0)
    runtime.device(0)

  def testLoadPackage(self):
    runtime = pygxtesting.new_graph_runtime()
    pkg = runtime.load(self.BASIC_PACKAGE)
    self.assertEqual(pkg.name(), "basic")
    self.assertEqual(pkg.fullname(), self.BASIC_PACKAGE)

  def testBuildPackage_NotFound(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    with self.assertRaisesRegex(RuntimeError, "gx/fake"):
      device.build_package("gx/fake")
    with self.assertRaisesRegex(cgx.Error, "gx/fake"):
      device.build("gx/fake")
    self.assertEqual(str(cgx.Error(" \ta\tb\n")), "a  b")

  def testPackage_ListInterfaces(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    self.assertEqual(package.ir().name(), "basic")
    ifaces = package.list_interfaces()
    wants = [
        "Empty",
        "Basic",
    ]
    self.assertLen(ifaces, len(wants))
    for i, vr in enumerate(ifaces):
      self.assertEqual(vr.name(), wants[i])

  def testPackage_ListStaticVars(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.PKGVARS_PACKAGE)
    staticvars = package.list_static_vars()
    wants = [
        "Var1",
        "Size",
    ]
    self.assertLen(staticvars, len(wants))
    for i, vr in enumerate(staticvars):
      self.assertEqual(vr.name(), wants[i])

  def testFindStaticVar(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.PKGVARS_PACKAGE)
    self.assertTrue(package.has_static_var("Var1"))
    static_var = package.find_static_var("Var1")
    self.assertEqual(static_var.name(), "Var1")

  def testFindStaticVar_NotFound(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.PKGVARS_PACKAGE)
    self.assertFalse(package.has_static_var("Fake"))
    with self.assertRaises(RuntimeError):
      package.find_static_var("Fake")

  def testSetStaticVar(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.PKGVARS_PACKAGE)
    static_var = package.find_static_var("Var1")
    static_var.set_value(42)
    function = package.find_function("ReturnVar1")
    result = function.run([])
    values = result.return_values()
    self.assertEqual(values[0].int32_value(), 42)

  def testPackage_ListFunctions(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    funcs = package.list_functions()
    wants = [
        "ReturnFloat32",
        "ReturnArrayFloat32",
        "ReturnMultiple",
        "New",
        "AddPrivate",
    ]
    self.assertLen(funcs, len(wants))
    for i, fn in enumerate(funcs):
      self.assertEqual(fn.name(), wants[i])

  def testFindFunction(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    self.assertTrue(package.has_function("ReturnArrayFloat32"))
    function = package.find_function("ReturnArrayFloat32")
    self.assertEqual(function.name(), "ReturnArrayFloat32")
    self.assertEqual(
        function.doc(), "ReturnArrayFloat32 returns a float32 tensor.\n"
    )
    self.assertEqual(
        str(function),
        "func ReturnArrayFloat32() [2]float32 {\n\treturn [2]float32{4.2,"
        " 42}\n}",
    )
    self.assertEqual(function.num_params(), 0)

  def testFindFunction_Signature(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.PARAMETERS_PACKAGE)
    function = package.find_function("AddInt")
    signature = function.signature()
    self.assertLen(signature.parameters(), 2)
    self.assertEqual(signature.parameters()[0].name, "x")
    self.assertEqual(signature.parameters()[0].kind, cgx.Kind.CGX_INT64)
    self.assertEqual(signature.parameters()[1].name, "y")
    self.assertEqual(signature.parameters()[1].kind, cgx.Kind.CGX_INT64)
    self.assertLen(signature.results(), 1)
    self.assertEqual(signature.results()[0].name, "")
    self.assertEqual(signature.results()[0].kind, cgx.Kind.CGX_INT64)

  def testFindFunction_NotFound(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    self.assertFalse(package.has_function("Fake"))
    with self.assertRaises(RuntimeError):
      package.find_function("Fake")

  def testFindInterface_NotFound(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    with self.assertRaises(RuntimeError):
      package.find_interface("Complicated")

  def testRunFunction_ReturnFloat32(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    function = package.find_function("ReturnFloat32")
    result = function.run([])

    values = result.return_values()
    self.assertLen(values, 1)
    self.assertEqual(values[0].kind(), cgx.Kind.CGX_FLOAT32)
    self.assertAlmostEqual(values[0].float32_value(), 4.2, places=6)

  def testRunFunction_ReturnArrayFloat32(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    function = package.find_function("ReturnArrayFloat32")
    result = function.run([])

    values = result.return_values()
    self.assertLen(values, 1)
    self.assertEqual(values[0].kind(), cgx.Kind.CGX_ARRAY)
    self.assertEqual(str(values[0].get_type(package)), "[2]float32")
    self.assertEqual(
        str(values[0]),
        "DeviceArray{DeviceID: 0}: [2]float32{4.2, 42}",
    )

    shape = values[0].shape()
    self.assertEqual(shape.size(), 2)
    self.assertSequenceEqual(shape.axes(), [2])

    array = values[0].to_ndarray()
    self.assertTrue(array.flags.writeable)
    np.testing.assert_array_equal(array, np.array([4.2, 42], dtype=np.float32))

  def testRunFunction_ReturnMultiple(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    function = package.find_function("ReturnMultiple")
    result = function.run([])

    values = result.return_values()
    self.assertLen(values, 3)
    self.assertEqual(values[0].kind(), cgx.Kind.CGX_INT32)
    self.assertAlmostEqual(values[0].int32_value(), 0, places=6)
    self.assertEqual(values[1].kind(), cgx.Kind.CGX_FLOAT32)
    self.assertAlmostEqual(values[1].float32_value(), 1, places=6)
    self.assertEqual(values[2].kind(), cgx.Kind.CGX_FLOAT64)
    self.assertAlmostEqual(values[2].float64_value(), 2.71828, places=6)

  def testRunFunction_AddInt(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.PARAMETERS_PACKAGE)
    function = package.find_function("AddInt")
    self.assertEqual(function.num_params(), 2)
    self.assertEqual(function.param_dtype(0), cgx.Kind.CGX_INT64)
    self.assertEqual(function.param_dtype(1), cgx.Kind.CGX_INT64)
    result = function.run(
        [
            cgx.Value.from_int64(device, 100),
            cgx.Value.from_int64(device, 200),
        ],
    )

    values = result.return_values()
    self.assertLen(values, 1)
    self.assertEqual(values[0].kind(), cgx.Kind.CGX_INT64)
    self.assertEqual(values[0].int64_value(), 300)

  def testRunFunction_MissingParameters(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.PARAMETERS_PACKAGE)
    function = package.find_function("AddInt")
    with self.assertRaises(RuntimeError):
      function.run([])

  def testNewShape_Float32Array(self):
    shape = cgx.Shape.new(cgx.Kind.CGX_FLOAT32, (3, 4))
    self.assertEqual(shape.size(), 3 * 4)
    self.assertSequenceEqual(shape.axes(), (3, 4))
    self.assertEqual(shape.element_kind(), cgx.Kind.CGX_FLOAT32)

  def testNewValue_FromNdarray(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    data = np.array([4.2, 42], dtype=np.float32)
    value = cgx.Value.from_ndarray(device, data)

    shape = value.shape()
    self.assertEqual(shape.size(), 2)
    self.assertSequenceEqual(shape.axes(), [2])

    np.testing.assert_array_equal(value.to_ndarray(), data)

  def testRunMethod_AddPrivate(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    new_func = package.find_function("New")
    new_result = new_func.run([]).return_values()
    basic_interface = new_result[0].get_type(package)
    self.assertIsNotNone(basic_interface)
    self.assertEqual(self.BASIC_PACKAGE, basic_interface.package_name())
    self.assertEqual("Basic", basic_interface.name())
    add_func = basic_interface.find_method("AddPrivate")
    add_result = add_func.run_method(new_result[0], []).return_values()
    self.assertEqual(add_result[0].int32_value(), 6)

  def testRunMethod_FindInterface(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    basic_interface = package.find_interface("Basic")
    new_func = package.find_function("New")
    new_result = new_func.run([]).return_values()
    self.assertIsNotNone(basic_interface)
    add_func = basic_interface.find_method("AddPrivate")
    add_result = add_func.run_method(new_result[0], []).return_values()
    self.assertEqual(add_result[0].int32_value(), 6)

  def testInterface_ListMethods(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    basic = package.find_interface("Basic")
    funcs = basic.list_methods()
    wants = ["Add", "AddPrivate", "SetFloat"]
    self.assertLen(funcs, len(wants))
    for i, fn in enumerate(funcs):
      self.assertEqual(fn.name(), wants[i])

  def testStruct_GetField(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    new_func = package.find_function("New")
    new_result = new_func.run([]).return_values()
    struct_value = new_result[0]
    self.assertIsNotNone(struct_value)
    value = struct_value.get_field("Int")
    self.assertEqual(value.kind(), cgx.Kind.CGX_INT32)
    self.assertEqual(value.int32_value(), 42)

  def testStruct_SetField(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    new_func = package.find_function("New")
    new_result = new_func.run([]).return_values()
    struct_value = new_result[0]
    self.assertIsNotNone(struct_value)
    struct_value.set_field("Int", cgx.Value.from_int32(device, 24))
    value = struct_value.get_field("Int")
    self.assertEqual(value.kind(), cgx.Kind.CGX_INT32)
    self.assertEqual(value.int32_value(), 24)

  def testStruct_ListFields(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.BASIC_PACKAGE)
    new_func = package.find_function("New")
    new_result = new_func.run([]).return_values()
    struct_value = new_result[0]
    self.assertIsNotNone(struct_value)
    fields = struct_value.get_type(package).list_fields()
    self.assertEqual(fields, ["Int", "Float", "Array"])

  def testUnexportedStructure(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.UNEXPORTED_PACKAGE)
    unexported_struct = package.find_function("New").run([]).return_values()[0]
    method_a = unexported_struct.get_type(package).find_method("A")
    got = method_a.run_method(unexported_struct, []).return_values()[0]
    self.assertEqual(got.float32_value(), 42)

  def testPyObjectWrapUnwrap(self):
    obj = {"hello": [1, 2, 3]}
    val = cgx.Value.from_object(obj)
    self.assertEqual(val.kind(), cgx.Kind.CGX_UNKNOWN)
    before_refs = sys.getrefcount(obj)
    self.assertIs(val.unwrap_pyobject(), obj)
    self.assertEqual(sys.getrefcount(obj), before_refs)

  def testPyObjectWrapUnwrap_RepeatedUnwrapPreservesRefCount(self):
    obj = {"hello": [1, 2, 3]}
    val = cgx.Value.from_object(obj)
    before_refs = sys.getrefcount(obj)
    for _ in range(10):
      unwrapped = val.unwrap_pyobject()
      self.assertIs(unwrapped, obj)
      del unwrapped
    self.assertEqual(sys.getrefcount(obj), before_refs)

  def testPyObjectWrapUnwrap_OutlivesValue(self):
    obj = {"hello": [1, 2, 3]}
    val = cgx.Value.from_object(obj)
    unwrapped = val.unwrap_pyobject()
    del val, obj
    gc.collect()
    cgx.run_go_gc()
    self.assertEqual(unwrapped, {"hello": [1, 2, 3]})

  def testPyObjectUnwrap_NonObjectValue(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    val = cgx.Value.from_int64(device, 42)
    with self.assertRaisesRegex(ValueError, "does not wrap a Python object"):
      val.unwrap_pyobject()

  def testPyObjectUnwrap_InvalidHandle(self):
    with self.assertRaisesRegex(ValueError, "does not wrap a Python object"):
      cgx.Value(0, owned=False).unwrap_pyobject()
    val = cgx.Value.from_object({"hello": [1, 2, 3]})
    stale = cgx.Value(val.handle, owned=False)
    del val
    gc.collect()
    with self.assertRaisesRegex(ValueError, "does not wrap a Python object"):
      stale.unwrap_pyobject()

  def testPyObjectUnwrap_NilPyObject(self):
    val = cgx.Value(_c_lib.pygx_wrap_py_object(None))
    with self.assertRaisesRegex(ValueError, "wrapped Python object is nil"):
      val.unwrap_pyobject()

  def testValueFromPyObj_UnknownKind(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    wrapped_int = cgx.value_from_pyobj(device, 42, cgx.Kind.CGX_UNKNOWN)
    self.assertEqual(wrapped_int.unwrap_pyobject(), 42)
    wrapped_none = cgx.value_from_pyobj(device, None, cgx.Kind.CGX_UNKNOWN)
    self.assertIsNone(wrapped_none.unwrap_pyobject())

  def testValueFromPyObj_NoneAsDynamicDim(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    none_int = cgx.value_from_pyobj(device, None, cgx.Kind.CGX_INT)
    self.assertEqual(none_int.int64_value(), -1)
    none_list = cgx.value_from_pyobj(device, [None, 4], cgx.Kind.CGX_INT)
    np.testing.assert_array_equal(none_list.to_ndarray(), np.array([-1, 4]))

  def testStaticBuilder(self):
    bld = cgx.static_builder()
    self.assertNotEqual(bld.handle, 0)

  def testTraceCallback(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    package = device.build_package(self.TRACE_PACKAGE)
    fn = package.find_function("TraceInt")
    message_from_trace = None
    value_from_trace = None

    def trace(*args):
      nonlocal message_from_trace, value_from_trace
      message_from_trace = args[0]
      value_from_trace = args[1].to_host()

    cgx.set_trace_callback(trace)
    try:
      fn.run([cgx.Value.from_int64(device, 42)])
    finally:
      importlib.reload(cgx)

    self.assertEqual(message_from_trace, "hello")
    self.assertEqual(value_from_trace, 42)

    with mock.patch("builtins.print") as mock_print:
      fn.run([cgx.Value.from_int64(device, 42)])
    self.assertEqual(mock_print.call_count, 2)

  def testHandleRelease(self):
    runtime = pygxtesting.new_graph_runtime()
    device = runtime.get_device(0)
    gc.collect()
    before = cgx.handle_count()
    for _ in range(20):
      v = cgx.Value.from_float32(device, 1.25)
      _ = v.float32_value()
    arr_val = cgx.Value.from_ndarray(
        device, np.array([1.0, 2.0], dtype=np.float32)
    )
    arr = arr_val.to_ndarray()
    with self.assertRaises(RuntimeError):
      device.build_package("gx/fake")
    del v, arr, arr_val
    gc.collect()
    after = cgx.handle_count()
    self.assertEqual(before, after)


if __name__ == "__main__":
  googletest.main()
