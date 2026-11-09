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

package pyelts

// #include "Python.h"
// #include "third_party/gxlang/gx/golang/binder/cgx/cgx.h"
import "C"

import (
	"unsafe"

	"github.com/gx-org/gx/cgx/handle"
	"github.com/gx-org/pygx/internal/gopy"
)

// pygx_wrap_py_object returns a new GX value from a Python object.
//
// cobj is passed as a borrowed reference. gopy.Wrap increments its Python
// reference count (Py_IncRef under the GIL) so the gopy.Object held by Element
// owns a strong reference, and registers a Go finalizer to decrement it
// (Py_DecRef). handle.Wrap pins the Element in the cgx handle table so Go's
// garbage collector keeps it alive while the opaque cgx_handle is held outside
// Go. Releasing the handle via cgx_release_reference only removes the Element
// from the handle table (other GX values in Go may still reference it); the
// Python reference count is decremented once Element.Finalize is called or the
// Go garbage collector runs the gopy.Object finalizer.
//
//export pygx_wrap_py_object
func pygx_wrap_py_object(cobj *C.PyObject) C.cgx_handle {
	el := New(gopy.Wrap(unsafe.Pointer(cobj)))
	return C.cgx_handle(handle.Wrap(el))
}

// pygx_unwrap_py_object returns the original pyobject from the GX value.
//
//export pygx_unwrap_py_object
func pygx_unwrap_py_object(h C.cgx_handle) *C.PyObject {
	el := handle.Unwrap[*Element](handle.Handle(h))
	return (*C.PyObject)(el.PyObject().Ptr())
}
