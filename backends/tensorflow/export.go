// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tensorflow

// #include "third_party/gxlang/gx/golang/binder/cgx/cgx.h"
import "C"

import (
	"github.com/gx-org/gx/api"
	"github.com/gx-org/gx/cgx/handle"
	"github.com/gx-org/pygx"
)

// gxtf_runtime returns a new GX runtime with a TensorFlow backend.
//
// The function is called from Python using ctypes
// (see third_party/gxlang/pygx/backends/tensorflow/gxtf.py).
//
//export gxtf_runtime
func gxtf_runtime() C.struct_cgx_runtime_new_result {
	rtm := New(pygx.Builder())
	return C.struct_cgx_runtime_new_result{
		runtime: C.cgx_runtime(handle.Wrap[*api.Runtime](rtm)),
	}
}
