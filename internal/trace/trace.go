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

// Package trace registers a callback forward GX values to Python for trace.
package trace

// #include "Python.h"
import "C"

import (
	"unsafe"

	"github.com/gx-org/gx/api/hostio"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/golang/binder/gobindings/core"
	"github.com/gx-org/pygx/internal/gopy"
	"github.com/gx-org/pygx/internal/pyelts"
)

type callback struct {
	fun *gopy.Object
}

// SetTraceCallback sets a Python callback for GX trace.
//
//export pygx_set_trace_callback
func pygx_set_trace_callback(cObj *C.PyObject) {
	cb := &callback{
		fun: gopy.Wrap(unsafe.Pointer(cObj)),
	}
	core.DefaultTracer = cb
}

func (cb *callback) Trace(file *ir.File, call *ir.FuncCallExpr, values []hostio.Value) error {
	pyvals := make([]any, len(values))
	for i, val := range values {
		var err error
		pyvals[i], err = pyelts.ObjectFromValue(val)
		if err != nil {
			return err
		}
	}
	_, err := cb.fun.Call(pyvals...)
	return err
}
