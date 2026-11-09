"""GX runtime using Tensorflow as a backend."""

import ctypes

from google3.third_party.gxlang.pygx import cgx

verbose_error = cgx.verbose_error

_CgxHandle = ctypes.c_uint64


class _CgxRuntimeNewResult(ctypes.Structure):
  """Mirrors struct cgx_runtime_new_result defined in cgx.h."""

  _fields_ = [
      ("runtime", _CgxHandle),
      ("error", _CgxHandle),
  ]


# gxtf_runtime is exported by the TensorFlow backend Go library
# (//third_party/gxlang/pygx/backends/tensorflow:tensorflow) which, like the
# rest of the cgx C API, is statically linked into the binary.
_c_lib = ctypes.CDLL(None)

_c_lib.gxtf_runtime.argtypes = []
_c_lib.gxtf_runtime.restype = _CgxRuntimeNewResult


def new_runtime() -> cgx.Runtime:
  """Returns a new GX runtime with a Tensorflow backend."""
  res = _c_lib.gxtf_runtime()
  cgx.check_error(res.error)
  return cgx.Runtime(res.runtime)


def set_as_default():
  """Set the Tensorflow and static loader as the default Python GX runtime."""
  cgx.default_runtime = new_runtime()
