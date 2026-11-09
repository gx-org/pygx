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
	"strings"
	"sync"

	"github.com/pkg/errors"
)

type typeError struct {
	err error
}

func newTypeError(s string, args ...any) error {
	return typeError{err: errors.Errorf(s, args...)}
}

func (err typeError) Error() string {
	return err.err.Error()
}

// formattingThreads tracks the Python threads currently inside formatTraceback
// to prevent infinite recursion if formatting a traceback itself raises a
// Python exception (which calls back into getPythonError). Keying by
// *C.PyThreadState keeps the guard per thread so that a thread switch while
// running Python code in formatTraceback does not suppress tracebacks on other
// threads.
var (
	formattingMu      sync.Mutex
	formattingThreads = make(map[*C.PyThreadState]struct{})
)

func enterFormatTraceback(tstate *C.PyThreadState) bool {
	formattingMu.Lock()
	defer formattingMu.Unlock()
	if _, ok := formattingThreads[tstate]; ok {
		return false
	}
	formattingThreads[tstate] = struct{}{}
	return true
}

func leaveFormatTraceback(tstate *C.PyThreadState) {
	formattingMu.Lock()
	delete(formattingThreads, tstate)
	formattingMu.Unlock()
}

func getPythonError() error {
	var ptype, pvalue, ptraceback *C.PyObject
	C.PyErr_Fetch(&ptype, &pvalue, &ptraceback)
	if pvalue == nil {
		return nil
	}
	C.PyErr_NormalizeException(&ptype, &pvalue, &ptraceback)
	defer C.Py_DecRef(ptype)
	defer C.Py_DecRef(pvalue)
	defer C.Py_DecRef(ptraceback)
	tstate := C.PyThreadState_Get()
	if ptraceback == nil || !enterFormatTraceback(tstate) {
		return errors.Errorf("uncatch exception in Python:\n\n%s", cPyObjectToStr(pvalue))
	}
	errS, err := formatTraceback(ptype, pvalue, ptraceback)
	leaveFormatTraceback(tstate)
	if err != nil {
		errS = strings.Join([]string{
			cPyObjectToStr(pvalue),
			cPyObjectToStr(ptraceback),
			"exception processing error: " + err.Error()},
			"\n")
	}
	return errors.Errorf("uncatch exception in Python:\n\n%s", errS)
}
