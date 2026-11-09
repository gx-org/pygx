"""Expose GX core structures to Python using ctypes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import ctypes
import enum
from typing import Any, Protocol, TypeVar

# Import JAX so that all devices are registered with XLA.
import jax  # pylint: disable=unused-import
import numpy as np

_CgxHandle = ctypes.c_uint64


class Kind(enum.IntEnum):
  """Kind of a GX value."""

  CGX_INVALID = 0
  CGX_BOOL = 1
  CGX_BFLOAT16 = 2
  CGX_FLOAT32 = 3
  CGX_FLOAT64 = 4
  CGX_INT32 = 5
  CGX_INT64 = 6
  CGX_INT = 7
  CGX_UINT32 = 8
  CGX_UINT64 = 9
  CGX_STRING = 10
  CGX_ARRAY = 11
  CGX_SLICE = 12
  CGX_STRUCT = 13
  CGX_INTERFACE = 14
  CGX_UNKNOWN = 15


class _CgxPackageIrLoadResult(ctypes.Structure):
  _fields_ = [
      ("package", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxDeviceGetResult(ctypes.Structure):
  _fields_ = [
      ("device", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxListStaticsResult(ctypes.Structure):
  _fields_ = [
      ("statics", ctypes.POINTER(_CgxHandle)),
      ("num_statics", ctypes.c_int),
      ("error", _CgxHandle),
  ]


class _CgxListFunctionsResult(ctypes.Structure):
  _fields_ = [
      ("funcs", ctypes.POINTER(_CgxHandle)),
      ("num_functions", ctypes.c_int),
      ("error", _CgxHandle),
  ]


class _CgxListInterfacesResult(ctypes.Structure):
  _fields_ = [
      ("ifaces", ctypes.POINTER(_CgxHandle)),
      ("num_ifaces", ctypes.c_int),
      ("error", _CgxHandle),
  ]


class _CgxFunctionSignatureElement(ctypes.Structure):
  _fields_ = [
      ("name", ctypes.c_char_p),
      ("kind", ctypes.c_uint32),
  ]


class _CgxFunctionSignatureResult(ctypes.Structure):
  _fields_ = [
      ("parameter", ctypes.POINTER(_CgxFunctionSignatureElement)),
      ("parameter_size", ctypes.c_uint32),
      ("result", ctypes.POINTER(_CgxFunctionSignatureElement)),
      ("result_size", ctypes.c_uint32),
      ("error", _CgxHandle),
  ]


class _CgxStaticFindResult(ctypes.Structure):
  _fields_ = [
      ("static_var", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxTypeFindResult(ctypes.Structure):
  _fields_ = [
      ("iface", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxTypeListFieldsResult(ctypes.Structure):
  _fields_ = [
      ("fields", ctypes.POINTER(ctypes.c_char_p)),
      ("num_fields", ctypes.c_int),
  ]


class _CgxFunctionFindResult(ctypes.Structure):
  _fields_ = [
      ("function", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxFunctionRunResult(ctypes.Structure):
  _fields_ = [
      ("values", ctypes.POINTER(_CgxHandle)),
      ("value_size", ctypes.c_uint32),
      ("error", _CgxHandle),
  ]


class _CgxValueNewResult(ctypes.Structure):
  _fields_ = [
      ("value", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxValueHostBufferResult(ctypes.Structure):
  _fields_ = [
      ("buffer", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxValueFieldGetResult(ctypes.Structure):
  _fields_ = [
      ("value", _CgxHandle),
      ("error", _CgxHandle),
  ]


class _CgxShapeAxesResult(ctypes.Structure):
  _fields_ = [
      ("axis_lengths", ctypes.POINTER(ctypes.c_int64)),
      ("num_axes", ctypes.c_uint32),
      ("error", _CgxHandle),
  ]


_c_lib = ctypes.CDLL(None)
_py_lib = ctypes.PyDLL(None)

_c_lib.free.argtypes = [ctypes.c_void_p]
_c_lib.free.restype = None

_c_lib.cgx_release_reference.argtypes = [_CgxHandle]
_c_lib.cgx_release_reference.restype = ctypes.c_size_t

_c_lib.cgx_handle_count.argtypes = []
_c_lib.cgx_handle_count.restype = ctypes.c_int64

_c_lib.cgx_error_verbose.argtypes = [ctypes.c_bool]
_c_lib.cgx_error_verbose.restype = None

_c_lib.cgx_error_message.argtypes = [_CgxHandle]
_c_lib.cgx_error_message.restype = ctypes.c_void_p

_c_lib.cgx_runtime_check.argtypes = [_CgxHandle]
_c_lib.cgx_runtime_check.restype = _CgxHandle

_c_lib.cgx_runtime_release.argtypes = [_CgxHandle]
_c_lib.cgx_runtime_release.restype = _CgxHandle

_c_lib.cgx_device_get.argtypes = [_CgxHandle, ctypes.c_int64]
_c_lib.cgx_device_get.restype = _CgxDeviceGetResult

_c_lib.cgx_device_get_runtime.argtypes = [_CgxHandle]
_c_lib.cgx_device_get_runtime.restype = _CgxHandle

_c_lib.cgx_package_ir_load.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_package_ir_load.restype = _CgxPackageIrLoadResult

_c_lib.cgx_package_ir_build_for.argtypes = [_CgxHandle, _CgxHandle]
_c_lib.cgx_package_ir_build_for.restype = _CgxHandle

_c_lib.cgx_package_ir_name.argtypes = [_CgxHandle]
_c_lib.cgx_package_ir_name.restype = ctypes.c_void_p

_c_lib.cgx_package_ir_fullname.argtypes = [_CgxHandle]
_c_lib.cgx_package_ir_fullname.restype = ctypes.c_void_p

_c_lib.cgx_package_get_ir.argtypes = [_CgxHandle]
_c_lib.cgx_package_get_ir.restype = _CgxHandle

_c_lib.cgx_package_list_statics.argtypes = [_CgxHandle]
_c_lib.cgx_package_list_statics.restype = _CgxListStaticsResult

_c_lib.cgx_free_list_statics_result.argtypes = [
    ctypes.POINTER(_CgxListStaticsResult)
]
_c_lib.cgx_free_list_statics_result.restype = None

_c_lib.cgx_static_has.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_static_has.restype = ctypes.c_bool

_c_lib.cgx_static_find.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_static_find.restype = _CgxStaticFindResult

_c_lib.cgx_static_name.argtypes = [_CgxHandle]
_c_lib.cgx_static_name.restype = ctypes.c_void_p

_c_lib.cgx_static_set.argtypes = [_CgxHandle, ctypes.c_int64]
_c_lib.cgx_static_set.restype = _CgxHandle

_c_lib.cgx_package_list_functions.argtypes = [_CgxHandle]
_c_lib.cgx_package_list_functions.restype = _CgxListFunctionsResult

_c_lib.cgx_free_list_functions_result.argtypes = [
    ctypes.POINTER(_CgxListFunctionsResult)
]
_c_lib.cgx_free_list_functions_result.restype = None

_c_lib.cgx_package_list_interfaces.argtypes = [_CgxHandle]
_c_lib.cgx_package_list_interfaces.restype = _CgxListInterfacesResult

_c_lib.cgx_free_list_interfaces_result.argtypes = [
    ctypes.POINTER(_CgxListInterfacesResult)
]
_c_lib.cgx_free_list_interfaces_result.restype = None

_c_lib.cgx_type_find.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_type_find.restype = _CgxTypeFindResult

_c_lib.cgx_type_method_find.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_type_method_find.restype = _CgxFunctionFindResult

_c_lib.cgx_type_name.argtypes = [_CgxHandle]
_c_lib.cgx_type_name.restype = ctypes.c_void_p

_c_lib.cgx_type_string.argtypes = [_CgxHandle]
_c_lib.cgx_type_string.restype = ctypes.c_void_p

_c_lib.cgx_type_package_name.argtypes = [_CgxHandle]
_c_lib.cgx_type_package_name.restype = ctypes.c_void_p

_c_lib.cgx_type_list_fields.argtypes = [_CgxHandle]
_c_lib.cgx_type_list_fields.restype = _CgxTypeListFieldsResult

_c_lib.cgx_free_type_list_fields_result.argtypes = [
    ctypes.POINTER(_CgxTypeListFieldsResult)
]
_c_lib.cgx_free_type_list_fields_result.restype = None

_c_lib.cgx_type_list_methods.argtypes = [_CgxHandle]
_c_lib.cgx_type_list_methods.restype = _CgxListFunctionsResult

_c_lib.cgx_function_has.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_function_has.restype = ctypes.c_bool

_c_lib.cgx_function_find.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_function_find.restype = _CgxFunctionFindResult

_c_lib.cgx_function_run.argtypes = [
    _CgxHandle,
    _CgxHandle,
    ctypes.c_int,
    ctypes.POINTER(_CgxHandle),
]
_c_lib.cgx_function_run.restype = _CgxFunctionRunResult

_c_lib.cgx_free_function_run_result.argtypes = [
    ctypes.POINTER(_CgxFunctionRunResult)
]
_c_lib.cgx_free_function_run_result.restype = None

_c_lib.cgx_function_name.argtypes = [_CgxHandle]
_c_lib.cgx_function_name.restype = ctypes.c_void_p

_c_lib.cgx_function_string.argtypes = [_CgxHandle]
_c_lib.cgx_function_string.restype = ctypes.c_void_p

_c_lib.cgx_function_doc.argtypes = [_CgxHandle]
_c_lib.cgx_function_doc.restype = ctypes.c_void_p

_c_lib.cgx_function_signature.argtypes = [_CgxHandle]
_c_lib.cgx_function_signature.restype = _CgxFunctionSignatureResult

_c_lib.cgx_free_function_signature_result.argtypes = [
    ctypes.POINTER(_CgxFunctionSignatureResult)
]
_c_lib.cgx_free_function_signature_result.restype = None

_c_lib.cgx_function_num_params.argtypes = [_CgxHandle]
_c_lib.cgx_function_num_params.restype = ctypes.c_int64

_c_lib.cgx_function_param_dtype.argtypes = [_CgxHandle, ctypes.c_int64]
_c_lib.cgx_function_param_dtype.restype = ctypes.c_uint32

_c_lib.cgx_value_new_bool.argtypes = [_CgxHandle, ctypes.c_bool]
_c_lib.cgx_value_new_bool.restype = _CgxValueNewResult

_c_lib.cgx_value_new_float32.argtypes = [_CgxHandle, ctypes.c_float]
_c_lib.cgx_value_new_float32.restype = _CgxValueNewResult

_c_lib.cgx_value_new_float64.argtypes = [_CgxHandle, ctypes.c_double]
_c_lib.cgx_value_new_float64.restype = _CgxValueNewResult

_c_lib.cgx_value_new_int32.argtypes = [_CgxHandle, ctypes.c_int32]
_c_lib.cgx_value_new_int32.restype = _CgxValueNewResult

_c_lib.cgx_value_new_int64.argtypes = [_CgxHandle, ctypes.c_int64]
_c_lib.cgx_value_new_int64.restype = _CgxValueNewResult

_c_lib.cgx_value_new_uint32.argtypes = [_CgxHandle, ctypes.c_uint32]
_c_lib.cgx_value_new_uint32.restype = _CgxValueNewResult

_c_lib.cgx_value_new_uint64.argtypes = [_CgxHandle, ctypes.c_uint64]
_c_lib.cgx_value_new_uint64.restype = _CgxValueNewResult

_c_lib.cgx_value_send.argtypes = [
    _CgxHandle,
    _CgxHandle,
    ctypes.c_void_p,
    ctypes.c_uint64,
]
_c_lib.cgx_value_send.restype = _CgxValueNewResult

_c_lib.cgx_value_kind_of.argtypes = [_CgxHandle]
_c_lib.cgx_value_kind_of.restype = ctypes.c_uint32

_c_lib.cgx_value_shape.argtypes = [_CgxHandle]
_c_lib.cgx_value_shape.restype = _CgxHandle

_c_lib.cgx_value_get_bool.argtypes = [_CgxHandle]
_c_lib.cgx_value_get_bool.restype = ctypes.c_bool

_c_lib.cgx_value_get_float32.argtypes = [_CgxHandle]
_c_lib.cgx_value_get_float32.restype = ctypes.c_float

_c_lib.cgx_value_get_float64.argtypes = [_CgxHandle]
_c_lib.cgx_value_get_float64.restype = ctypes.c_double

_c_lib.cgx_value_get_int32.argtypes = [_CgxHandle]
_c_lib.cgx_value_get_int32.restype = ctypes.c_int32

_c_lib.cgx_value_get_int64.argtypes = [_CgxHandle]
_c_lib.cgx_value_get_int64.restype = ctypes.c_int64

_c_lib.cgx_value_get_uint32.argtypes = [_CgxHandle]
_c_lib.cgx_value_get_uint32.restype = ctypes.c_uint32

_c_lib.cgx_value_get_uint64.argtypes = [_CgxHandle]
_c_lib.cgx_value_get_uint64.restype = ctypes.c_uint64

_c_lib.cgx_value_host_buffer.argtypes = [_CgxHandle]
_c_lib.cgx_value_host_buffer.restype = _CgxValueHostBufferResult

_c_lib.cgx_value_field_set.argtypes = [
    _CgxHandle,
    ctypes.c_char_p,
    _CgxHandle,
]
_c_lib.cgx_value_field_set.restype = _CgxHandle

_c_lib.cgx_value_field_get.argtypes = [_CgxHandle, ctypes.c_char_p]
_c_lib.cgx_value_field_get.restype = _CgxValueFieldGetResult

_c_lib.cgx_value_get_type.argtypes = [_CgxHandle, _CgxHandle]
_c_lib.cgx_value_get_type.restype = _CgxHandle

_c_lib.cgx_value_string.argtypes = [_CgxHandle]
_c_lib.cgx_value_string.restype = ctypes.c_void_p

_c_lib.cgx_shape_new.argtypes = [
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
]
_c_lib.cgx_shape_new.restype = _CgxHandle

_c_lib.cgx_shape_axes.argtypes = [_CgxHandle]
_c_lib.cgx_shape_axes.restype = _CgxShapeAxesResult

_c_lib.cgx_free_shape_axes_result.argtypes = [
    ctypes.POINTER(_CgxShapeAxesResult)
]
_c_lib.cgx_free_shape_axes_result.restype = None

_c_lib.cgx_shape_size.argtypes = [_CgxHandle]
_c_lib.cgx_shape_size.restype = ctypes.c_int

_c_lib.cgx_shape_element_kind.argtypes = [_CgxHandle]
_c_lib.cgx_shape_element_kind.restype = ctypes.c_uint32

_c_lib.cgx_host_buffer_acquire_data.argtypes = [_CgxHandle]
_c_lib.cgx_host_buffer_acquire_data.restype = ctypes.c_void_p

_c_lib.cgx_host_buffer_release_data.argtypes = [_CgxHandle, ctypes.c_void_p]
_c_lib.cgx_host_buffer_release_data.restype = None

_c_lib.cgx_run_gc.argtypes = []
_c_lib.cgx_run_gc.restype = None

_c_lib.cgx_new_embed_builder.argtypes = []
_c_lib.cgx_new_embed_builder.restype = _CgxHandle

_py_lib.pygx_wrap_py_object.argtypes = [ctypes.py_object]
_py_lib.pygx_wrap_py_object.restype = _CgxHandle

_py_lib.pygx_unwrap_py_object.argtypes = [_CgxHandle]
_py_lib.pygx_unwrap_py_object.restype = ctypes.py_object

_py_lib.pygx_set_trace_callback.argtypes = [ctypes.py_object]
_py_lib.pygx_set_trace_callback.restype = None


def from_heap_cstring(ptr: int) -> str:
  """Converts a heap-allocated C string to a Python str and frees the pointer."""
  try:
    return ctypes.string_at(ptr).decode("utf-8")
  finally:
    _c_lib.free(ptr)


def check_error(err: int, error_type: type[Exception] = RuntimeError) -> None:
  """Raises an error_type exception if err is a non-zero cgx_error handle."""
  if not err:
    return
  msg = from_heap_cstring(_c_lib.cgx_error_message(err))
  _c_lib.cgx_release_reference(err)
  raise error_type(msg)


class _ResultWithError(Protocol):
  error: int


_ResultT = TypeVar("_ResultT", bound=_ResultWithError)


def _check_result(res: _ResultT) -> _ResultT:
  """Raises a RuntimeError if res.error is non-zero, otherwise returns res."""
  check_error(res.error)
  return res


class Handle:
  """Base wrapper around an opaque cgx_handle.

  An owned handle is released when its wrapper is destroyed.
  """

  _handle: int = 0
  _owned: bool = False

  def __init__(self, handle: int, owned: bool = True):
    self._handle = handle
    self._owned = owned

  @property
  def handle(self) -> int:
    return self._handle

  def _release(self) -> None:
    if self._owned and self._handle:
      _c_lib.cgx_release_reference(self._handle)
      self._handle = 0

  def __del__(self) -> None:
    self._release()


def handle_count() -> int:
  """Returns the number of outstanding CGX handles."""
  return _c_lib.cgx_handle_count()


def set_trace_callback(cb: Callable[..., None]) -> None:
  """Sets the Python callback invoked by GX trace()."""
  _py_lib.pygx_set_trace_callback(cb)


def verbose_error(enable: bool) -> None:
  """Enables or disables verbose GX error messages."""
  _c_lib.cgx_error_verbose(enable)


# Default GX runtime used by default when importing modules
# using python import directive.
default_runtime: Runtime | None = None


def default_trace(*vals: Any) -> None:
  for val in vals:
    print(val)


set_trace_callback(default_trace)


def run_go_gc() -> None:
  """Run the Go garbage collector."""
  _c_lib.cgx_run_gc()


class Builder(Handle):
  """GX Builder."""


class Error(Exception):

  def __init__(self, msg: str):
    msg = msg.strip()
    msg = msg.replace("\t", "  ")
    super().__init__(msg)


class PackageIR(Handle):
  """GX Package IR."""

  def name(self) -> str:
    return from_heap_cstring(_c_lib.cgx_package_ir_name(self._handle))

  def fullname(self) -> str:
    return from_heap_cstring(_c_lib.cgx_package_ir_fullname(self._handle))

  def build_for(self, device: Device) -> Package:
    """Compiles the package for a device."""
    return Package(_c_lib.cgx_package_ir_build_for(self._handle, device.handle))


class StaticVar(Handle):
  """GX Static Variable."""

  def name(self) -> str:
    return from_heap_cstring(_c_lib.cgx_static_name(self._handle))

  def set_value(self, val: int) -> None:
    check_error(_c_lib.cgx_static_set(self._handle, val))


class FunctionSignatureField:
  """Describes a parameter or result field in a GX FunctionSignature."""

  def __init__(self, name: str, kind: Kind):
    self.name = name
    self.kind = kind


class FunctionSignature:
  """Signature of a GX function."""

  def __init__(
      self,
      parameters: list[FunctionSignatureField],
      results: list[FunctionSignatureField],
  ):
    self._parameters = parameters
    self._results = results

  def parameters(self) -> list[FunctionSignatureField]:
    return self._parameters

  def results(self) -> list[FunctionSignatureField]:
    return self._results


class FunctionResult:
  """Result of running a GX function."""

  def __init__(self, values: list[Value]):
    self._values = values

  def return_values(self) -> list[Value]:
    return self._values


class Function(Handle):
  """Compiled GX function."""

  def _run(self, recv_handle: int, args: Sequence[Value]) -> FunctionResult:
    """Runs the function with an optional receiver handle and arguments."""
    num_args = len(args)
    c_args = (
        (_CgxHandle * num_args)(*(a.handle for a in args)) if num_args else None
    )
    res = _check_result(
        _c_lib.cgx_function_run(self._handle, recv_handle, num_args, c_args)
    )
    try:
      values = [Value(res.values[i]) for i in range(res.value_size)]
    finally:
      _c_lib.cgx_free_function_run_result(ctypes.byref(res))
    return FunctionResult(values)

  def run(self, args: Sequence[Value]) -> FunctionResult:
    return self._run(0, args)

  def run_method(
      self, receiver: Value, args: Sequence[Value]
  ) -> FunctionResult:
    """Runs the function or method with the given receiver and arguments."""
    return self._run(receiver.handle, args)

  def signature(self) -> FunctionSignature:
    """Returns the signature of the function."""
    res = _check_result(_c_lib.cgx_function_signature(self._handle))
    try:
      params = [
          FunctionSignatureField(
              res.parameter[i].name.decode("utf-8"),
              Kind(res.parameter[i].kind),
          )
          for i in range(res.parameter_size)
      ]
      results = [
          FunctionSignatureField(
              res.result[i].name.decode("utf-8"),
              Kind(res.result[i].kind),
          )
          for i in range(res.result_size)
      ]
    finally:
      _c_lib.cgx_free_function_signature_result(ctypes.byref(res))
    return FunctionSignature(params, results)

  def num_params(self) -> int:
    return _c_lib.cgx_function_num_params(self._handle)

  def param_dtype(self, arg: int) -> Kind:
    return Kind(_c_lib.cgx_function_param_dtype(self._handle, arg))

  def name(self) -> str:
    return from_heap_cstring(_c_lib.cgx_function_name(self._handle))

  def __str__(self) -> str:
    return from_heap_cstring(_c_lib.cgx_function_string(self._handle))

  def doc(self) -> str:
    return from_heap_cstring(_c_lib.cgx_function_doc(self._handle))


class Type(Handle):
  """GX Type."""

  def package_name(self) -> str:
    return from_heap_cstring(_c_lib.cgx_type_package_name(self._handle))

  def name(self) -> str:
    return from_heap_cstring(_c_lib.cgx_type_name(self._handle))

  def __str__(self) -> str:
    return from_heap_cstring(_c_lib.cgx_type_string(self._handle))

  def list_fields(self) -> list[str]:
    res = _c_lib.cgx_type_list_fields(self._handle)
    try:
      return [res.fields[i].decode("utf-8") for i in range(res.num_fields)]
    finally:
      _c_lib.cgx_free_type_list_fields_result(ctypes.byref(res))

  def find_method(self, name: str) -> Function:
    res = _check_result(
        _c_lib.cgx_type_method_find(self._handle, name.encode("utf-8"))
    )
    return Function(res.function)

  def list_methods(self) -> list[Function]:
    res = _check_result(_c_lib.cgx_type_list_methods(self._handle))
    try:
      return [Function(res.funcs[i]) for i in range(res.num_functions)]
    finally:
      _c_lib.cgx_free_list_functions_result(ctypes.byref(res))


class Package(Handle):
  """Compiled GX package for a device."""

  def ir(self) -> PackageIR:
    return PackageIR(_c_lib.cgx_package_get_ir(self._handle))

  def find_interface(self, name: str) -> Type:
    res = _check_result(
        _c_lib.cgx_type_find(self._handle, name.encode("utf-8"))
    )
    return Type(res.iface)

  def list_interfaces(self) -> list[Type]:
    res = _check_result(_c_lib.cgx_package_list_interfaces(self._handle))
    try:
      return [Type(res.ifaces[i]) for i in range(res.num_ifaces)]
    finally:
      _c_lib.cgx_free_list_interfaces_result(ctypes.byref(res))

  def has_static_var(self, name: str) -> bool:
    return _c_lib.cgx_static_has(self._handle, name.encode("utf-8"))

  def find_static_var(self, name: str) -> StaticVar:
    res = _check_result(
        _c_lib.cgx_static_find(self._handle, name.encode("utf-8"))
    )
    return StaticVar(res.static_var)

  def list_static_vars(self) -> list[StaticVar]:
    res = _check_result(_c_lib.cgx_package_list_statics(self._handle))
    try:
      return [StaticVar(res.statics[i]) for i in range(res.num_statics)]
    finally:
      _c_lib.cgx_free_list_statics_result(ctypes.byref(res))

  def has_function(self, name: str) -> bool:
    return _c_lib.cgx_function_has(self._handle, name.encode("utf-8"))

  def find_function(self, name: str) -> Function:
    res = _check_result(
        _c_lib.cgx_function_find(self._handle, name.encode("utf-8"))
    )
    return Function(res.function)

  def list_functions(self) -> list[Function]:
    res = _check_result(_c_lib.cgx_package_list_functions(self._handle))
    try:
      return [Function(res.funcs[i]) for i in range(res.num_functions)]
    finally:
      _c_lib.cgx_free_list_functions_result(ctypes.byref(res))


class Shape(Handle):
  """GX array shape."""

  @classmethod
  def new(cls, dtype: Kind, axis_lengths: Sequence[int]) -> Shape:
    num_axes = len(axis_lengths)
    if num_axes:
      c_axes = (ctypes.c_int64 * num_axes)(*axis_lengths)
    else:
      c_axes = None
    handle = _c_lib.cgx_shape_new(dtype, c_axes, num_axes)
    return cls(handle)

  def axes(self) -> list[int]:
    res = _check_result(_c_lib.cgx_shape_axes(self._handle))
    try:
      return [res.axis_lengths[i] for i in range(res.num_axes)]
    finally:
      _c_lib.cgx_free_shape_axes_result(ctypes.byref(res))

  def size(self) -> int:
    return _c_lib.cgx_shape_size(self._handle)

  def element_kind(self) -> Kind:
    return Kind(_c_lib.cgx_shape_element_kind(self._handle))


class _HostBuffer(Handle):
  """Host buffer backing a NumPy ndarray via __array_interface__."""

  _data: int | None = None

  def __init__(self, handle: int, shape: tuple[int, ...], dtype: np.dtype):
    super().__init__(handle)
    self._data = _c_lib.cgx_host_buffer_acquire_data(self._handle)
    self.__array_interface__ = {
        "version": 3,
        "shape": shape,
        "typestr": dtype.str,
        "data": (self._data, False),
    }

  def _release(self) -> None:
    if self._data and self._handle:
      # NOMUTANTS(SBR): Frees C/Go host buffer memory; unobservable in Python.
      _c_lib.cgx_host_buffer_release_data(self._handle, self._data)
      self._data = None
    super()._release()


class Value(Handle):
  """GX Value."""

  @classmethod
  def from_bool(cls, device: Device, val: bool) -> Value:
    res = _check_result(_c_lib.cgx_value_new_bool(device.handle, val))
    return cls(res.value)

  @classmethod
  def from_float32(cls, device: Device, val: float) -> Value:
    res = _check_result(_c_lib.cgx_value_new_float32(device.handle, val))
    return cls(res.value)

  @classmethod
  def from_float64(cls, device: Device, val: float) -> Value:
    res = _check_result(_c_lib.cgx_value_new_float64(device.handle, val))
    return cls(res.value)

  @classmethod
  def from_int32(cls, device: Device, val: int) -> Value:
    res = _check_result(_c_lib.cgx_value_new_int32(device.handle, val))
    return cls(res.value)

  @classmethod
  def from_int64(cls, device: Device, val: int) -> Value:
    res = _check_result(_c_lib.cgx_value_new_int64(device.handle, val))
    return cls(res.value)

  @classmethod
  def from_uint32(cls, device: Device, val: int) -> Value:
    res = _check_result(_c_lib.cgx_value_new_uint32(device.handle, val))
    return cls(res.value)

  @classmethod
  def from_uint64(cls, device: Device, val: int) -> Value:
    res = _check_result(_c_lib.cgx_value_new_uint64(device.handle, val))
    return cls(res.value)

  @classmethod
  def from_ndarray(cls, device: Device, input_array: np.ndarray) -> Value:
    """Transfers a NumPy ndarray to the given GX device."""
    arr = np.ascontiguousarray(input_array)
    kind = kind_from_numpy_dtype(arr.dtype)
    shape = Shape.new(kind, arr.shape)
    res = _check_result(
        _c_lib.cgx_value_send(
            device.handle,
            shape.handle,
            arr.ctypes.data_as(ctypes.c_void_p),
            arr.nbytes,
        )
    )
    return cls(res.value)

  @classmethod
  def from_object(cls, obj: object) -> Value:
    return cls(_py_lib.pygx_wrap_py_object(obj))

  def kind(self) -> Kind:
    return Kind(_c_lib.cgx_value_kind_of(self._handle))

  def shape(self) -> Shape:
    sh = _c_lib.cgx_value_shape(self._handle)
    if not sh:
      raise RuntimeError("value has no shape")
    return Shape(sh)

  def bool_value(self) -> bool:
    return _c_lib.cgx_value_get_bool(self._handle)

  def float32_value(self) -> float:
    return _c_lib.cgx_value_get_float32(self._handle)

  def float64_value(self) -> float:
    return _c_lib.cgx_value_get_float64(self._handle)

  def int32_value(self) -> int:
    return _c_lib.cgx_value_get_int32(self._handle)

  def int64_value(self) -> int:
    return _c_lib.cgx_value_get_int64(self._handle)

  def uint32_value(self) -> int:
    return _c_lib.cgx_value_get_uint32(self._handle)

  def uint64_value(self) -> int:
    return _c_lib.cgx_value_get_uint64(self._handle)

  def get_type(self, package: Package) -> Type:
    return Type(
        _c_lib.cgx_value_get_type(package.handle, self._handle)
    )

  def get_field(self, name: str) -> Value:
    res = _check_result(
        _c_lib.cgx_value_field_get(self._handle, name.encode("utf-8"))
    )
    return Value(res.value)

  def set_field(self, name: str, val: Value) -> None:
    check_error(
        _c_lib.cgx_value_field_set(
            self._handle, name.encode("utf-8"), val.handle
        )
    )

  def unwrap_pyobject(self) -> Any:
    """Returns the Python object wrapped by this GX value.

    Raises:
      ValueError: If this value does not wrap a live Python object.
    """
    return _py_lib.pygx_unwrap_py_object(self._handle)

  def to_ndarray(self) -> np.ndarray:
    shape = self.shape()
    axes = tuple(shape.axes())
    element_kind = shape.element_kind()
    dtype = _ATOMIC_KIND.get(element_kind)
    if dtype is None:
      raise TypeError(f"Unsupported array element kind: {element_kind}")
    res = _check_result(_c_lib.cgx_value_host_buffer(self._handle))
    host_buf = _HostBuffer(res.buffer, axes, np.dtype(dtype))
    return np.asarray(host_buf)

  def __str__(self) -> str:
    return from_heap_cstring(_c_lib.cgx_value_string(self._handle))

  def __repr__(self) -> str:
    return self.__str__()


class Device(Handle):
  """Device on a platform."""

  def __init__(self, handle: int, runtime: Runtime | None = None):
    super().__init__(handle)
    # The device keeps its runtime alive: the backend of the runtime is
    # finalized when the runtime is destroyed.
    self._runtime = runtime

  @property
  def raw(self) -> Device:
    return self

  def build_package(self, path: str) -> Package:
    """Loads and compiles the GX package at path for this device."""
    rtm = _c_lib.cgx_device_get_runtime(self._handle)
    try:
      res = _check_result(
          _c_lib.cgx_package_ir_load(rtm, path.encode("utf-8"))
      )
    finally:
      _c_lib.cgx_release_reference(rtm)
    try:
      pkg_handle = _c_lib.cgx_package_ir_build_for(res.package, self._handle)
    finally:
      _c_lib.cgx_release_reference(res.package)
    return Package(pkg_handle)

  def build(self, path: str) -> Package:
    try:
      return self.build_package(path)
    except Exception as err:  # pylint:disable=broad-exception-caught
      raise Error(str(err)) from err


class Runtime(Handle):
  """GX runtime.

  A runtime includes a platform, a graph builder, and a package importer.

  The backend of the runtime is finalized once the runtime and all its devices
  have been destroyed.
  """

  def __init__(self, handle: int):
    """Takes ownership of a raw cgx_runtime handle (e.g. obtained with ctypes).

    Args:
      handle: cgx_runtime handle.

    Raises:
      ValueError: if handle is not a valid handle to a GX runtime.
    """
    # cgx functions panic, which aborts the process, when given an invalid
    # handle: check the handle before taking ownership of it.
    check_error(_c_lib.cgx_runtime_check(handle), ValueError)
    super().__init__(handle)

  def _release(self) -> None:
    if self._owned and self._handle:
      # Errors cannot be reported when the Python object is destroyed.
      err = _c_lib.cgx_runtime_release(self._handle)
      if err:
        _c_lib.cgx_release_reference(err)
    super()._release()

  def get_device(self, num: int) -> Device:
    res = _check_result(_c_lib.cgx_device_get(self._handle, num))
    return Device(res.device, runtime=self)

  def device(self, num: int) -> Device:
    return self.get_device(num)

  def load(self, path: str) -> PackageIR:
    res = _check_result(
        _c_lib.cgx_package_ir_load(self._handle, path.encode("utf-8"))
    )
    return PackageIR(res.package)

  @property
  def raw(self) -> Runtime:
    return self


def to_host(val: Any) -> Any:
  """Transfer all the values to the host."""
  if not val:
    return None
  if hasattr(val, "to_host"):
    return val.to_host()
  return tuple(to_host(v) for v in val)


class Interface:
  """Interface provides methods given an interface and a value."""

  def __init__(self, dev: Device, iface: Type, receiver: Value):
    self.device = dev
    self.interface = iface
    self.receiver = receiver


class _Numerical:
  """GX numerical value, including arrays and scalars."""

  def __init__(self, val: Value):
    self._val = val

  def __str__(self):
    return str(self._val)

  def __repr__(self):
    return str(self._val)

  def kind(self) -> Kind:
    return self._val.kind()

  @property
  def value(self) -> Value:
    return self._val


class Array(_Numerical):
  """Array managed by GX."""

  def to_host(self) -> np.ndarray:
    return self._val.to_ndarray()


class Bool(_Numerical):
  """Bool atom managed by GX."""

  def to_host(self) -> bool:
    return self._val.bool_value()


class Float32(_Numerical):
  """Float32 atom managed by GX."""

  def to_host(self) -> float:
    return self._val.float32_value()


class Float64(_Numerical):
  """Float64 atom managed by GX."""

  def to_host(self) -> float:
    return self._val.float64_value()


class Int32(_Numerical):
  """Int32 atom managed by GX."""

  def to_host(self) -> int:
    return self._val.int32_value()


class Int64(_Numerical):
  """Int64 atom managed by GX."""

  def to_host(self) -> int:
    return self._val.int64_value()


class Int(_Numerical):
  """Int atom managed by GX."""

  def to_host(self) -> int:
    return self._val.int64_value()


class Uint32(_Numerical):
  """Uint32 atom managed by GX."""

  def to_host(self) -> int:
    return self._val.uint32_value()


class Uint64(_Numerical):
  """Uint64 atom managed by GX."""

  def to_host(self) -> int:
    return self._val.uint64_value()


def python_value_from_gx_value(
    device: Device | None, package: Package | None, val: Value
) -> _Numerical | Interface | str | Value:
  """Convert a GX value to a python value."""
  if package is not None and device is not None:
    valtype = val.get_type(package)
    if valtype.name():
      return interface_wrapper(Interface(device, valtype, val))

  match val.kind():
    case Kind.CGX_ARRAY:
      return Array(val)
    case Kind.CGX_BOOL:
      return Bool(val)
    case Kind.CGX_FLOAT32:
      return Float32(val)
    case Kind.CGX_FLOAT64:
      return Float64(val)
    case Kind.CGX_INT:
      return Int(val)
    case Kind.CGX_INT32:
      return Int32(val)
    case Kind.CGX_INT64:
      return Int64(val)
    case Kind.CGX_UINT32:
      return Uint32(val)
    case Kind.CGX_UINT64:
      return Uint64(val)
    case Kind.CGX_STRING:
      return str(val)
    case Kind.CGX_UNKNOWN:
      return val.unwrap_pyobject()
  return val


def kind_from_numpy_dtype(dtype: np.dtype) -> Kind:
  """Convert a numpy datatype into a GX kind."""
  match dtype:
    case np.bool:
      return Kind.CGX_BOOL
    case np.float32:
      return Kind.CGX_FLOAT32
    case np.float64:
      return Kind.CGX_FLOAT64
    case np.int32:
      return Kind.CGX_INT32
    case np.int64:
      return Kind.CGX_INT64
    case np.uint32:
      return Kind.CGX_UINT32
    case np.uint64:
      return Kind.CGX_UINT64
  return Kind.CGX_INVALID


_ATOMIC_KIND = {
    Kind.CGX_BOOL: np.bool,
    Kind.CGX_FLOAT32: np.float32,
    Kind.CGX_FLOAT64: np.float64,
    Kind.CGX_INT32: np.int32,
    Kind.CGX_INT64: np.int64,
    Kind.CGX_INT: np.int64,
    Kind.CGX_UINT32: np.uint32,
    Kind.CGX_UINT64: np.uint64,
}

# Sentinel value for an unknown/dynamic axis length, matching shapes.DynamicDim
# in GoMLX.
GOMLX_SHAPE_DYNAMIC_DIM = -1


def _gx_value_from_python_atomic(
    device: Device, val: Any, target: Kind
) -> Value:
  """Convert a Python atomic value into its corresponding value in GX."""
  match target:
    case Kind.CGX_BOOL:
      return Value.from_bool(device.raw, val)
    case Kind.CGX_FLOAT32:
      return Value.from_float32(device.raw, val)
    case Kind.CGX_FLOAT64:
      return Value.from_float64(device.raw, val)
    case Kind.CGX_INT32:
      return Value.from_int32(device.raw, val)
    case Kind.CGX_INT64:
      return Value.from_int64(device.raw, val)
    case Kind.CGX_INT:
      return Value.from_int64(device.raw, val)
    case Kind.CGX_UINT32:
      return Value.from_uint32(device.raw, val)
    case Kind.CGX_UINT64:
      return Value.from_uint64(device.raw, val)
  raise TypeError(
      f"cannot convert Python atomic {type(val)} to a GX value (target:"
      f" {target})"
  )


def _gx_value_from_numpy_array(
    device: Device, val: np.ndarray, target: Kind
) -> Value:
  val_kind = kind_from_numpy_dtype(val.dtype)
  if val_kind != target and not (
      val_kind == Kind.CGX_INT64 and target == Kind.CGX_INT
  ):
    raise TypeError(f"numpy array has {val_kind} data type but want {target}")
  return Value.from_ndarray(device.raw, val)


def value_from_pyobj(device: Device, val: Any, target: Kind) -> Value:
  """Convert a python value to a GX value."""
  if hasattr(val, "to_gx_value"):
    return val.to_gx_value()
  if isinstance(val, Value):
    return val
  if isinstance(val, _Numerical):
    return val.value
  if val is None and target == Kind.CGX_INT:
    val = GOMLX_SHAPE_DYNAMIC_DIM
  if not hasattr(val, "__len__") and target in _ATOMIC_KIND:
    return _gx_value_from_python_atomic(device, val, target)
  if isinstance(val, (list, tuple)) and target in _ATOMIC_KIND:
    if target == Kind.CGX_INT:
      val = [GOMLX_SHAPE_DYNAMIC_DIM if x is None else x for x in val]
    val = np.array(val, dtype=_ATOMIC_KIND[target])
  if isinstance(val, np.ndarray):
    return _gx_value_from_numpy_array(device, val, target)
  return Value.from_object(val)


def static_builder() -> Builder:
  """Returns a builder loading GX modules from source files embedded in the binary."""
  return Builder(_c_lib.cgx_new_embed_builder())


def interface_wrapper(iface: Interface) -> Interface:
  return iface
