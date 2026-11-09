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

package gopy

// #include "Python.h"
import "C"
import (
	"runtime"
)

// GIL stores the C reference to Python GIL state.
type GIL struct {
	gil C.PyGILState_STATE
}

// EnsureGILState locks the Python GIL and lock the Go routine into the current OS thread.
func EnsureGILState() *GIL {
	runtime.LockOSThread()
	return &GIL{C.PyGILState_Ensure()}
}

// ReleaseGILState unlocks the Python GIL and unlock the Go routine from the current OS thread.
func ReleaseGILState(gil *GIL) {
	C.PyGILState_Release(gil.gil)
	runtime.UnlockOSThread()
}

func gilIncRef(obj *C.PyObject) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	C.Py_IncRef(obj)
}

func gilDecRef(obj *C.PyObject) {
	// During Python interpreter shutdown, Go finalizers may run on background
	// goroutines after thread states have been cleared. Acquiring the GIL at
	// that point causes fatal assertion failures in PyThreadState_Clear.
	// We skip the DecRef if the interpreter is no longer initialized; the
	// process is exiting anyway so the leaked reference is harmless.
	if C.Py_IsInitialized() == 0 {
		return
	}
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	C.Py_DecRef(obj)
}
