# Copyright 2026 Google LLC
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

"""GX runtimes to test GX from Python."""

import ctypes

from google3.third_party.gxlang.pygx import cgx

_CgxHandle = ctypes.c_uint64


class _CgxRuntimeNewResult(ctypes.Structure):
  """Mirrors struct cgx_runtime_new_result defined in cgx.h."""

  _fields_ = [
      ("runtime", _CgxHandle),
      ("error", _CgxHandle),
  ]


# pygxtestdeps_graph_runtime is exported by the pygxtestdeps Go library
# (//third_party/gxlang/pygx/testing/pygxtestdeps) which, like the rest of the
# cgx C API, is statically linked into the binary.
_c_lib = ctypes.CDLL(None)

_c_lib.pygxtestdeps_graph_runtime.argtypes = []
_c_lib.pygxtestdeps_graph_runtime.restype = _CgxRuntimeNewResult


def new_graph_runtime() -> cgx.Runtime:
  """Returns a new GX runtime with a Go backend and the GX test packages.

  The backend of the runtime is finalized once the runtime and all its devices
  have been destroyed.
  """
  res = _c_lib.pygxtestdeps_graph_runtime()
  cgx.check_error(res.error)
  return cgx.Runtime(res.runtime)
