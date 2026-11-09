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

// Package gopy provides utilities for GX and Python to interact.
package gopy

// #include "Python.h"
// #include <stdlib.h>
//
// static inline int PyStringCheck(PyObject *ob) {
//   return PyUnicode_Check(ob);
// }
//
// static inline int PyLongCheck(PyObject *ob) {
//   return PyLong_Check(ob) && !PyBool_Check(ob);
// }
//
// static inline PyTypeObject* PyType(const PyObject *ob) {
//   return Py_TYPE(ob);
// }
//
// static inline const char* PyString_AsString(PyObject *ob) {
//   if (PyUnicode_Check(ob)) {
//     return PyUnicode_AsUTF8(ob);
//   }
//   return PyBytes_AsString(ob);
// }
import "C"

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/pkg/errors"
	"github.com/gomlx/compute/dtypes/bfloat16"
)

// Object encapsulates a reference to a Python object.
type Object struct {
	obj      *C.PyObject
	releaser func()
}

// Wrap a pointer to a Python object into an object.
func Wrap(ptr unsafe.Pointer) *Object {
	obj := (*C.PyObject)(ptr)
	gilIncRef(obj)
	return newObject(obj)
}

// WrapWithReleaser wraps a pointer into a Python object and set a callback for when the Python object is released.
func WrapWithReleaser(ptr unsafe.Pointer, rlsr func()) *Object {
	obj := Wrap(ptr)
	obj.releaser = rlsr
	return obj
}

func newObject(obj *C.PyObject) *Object {
	o := &Object{obj: obj}
	runtime.SetFinalizer(o, (*Object).Finalize)
	return o
}

// Finalize releases the reference to the Python object.
func (obj *Object) Finalize() {
	if obj == nil || obj.obj == nil {
		return
	}
	gilDecRef(obj.obj)
	obj.obj = nil
	if obj.releaser != nil {
		obj.releaser()
		obj.releaser = nil
	}
}

// Ptr returns the pointer to the Python object.
func (obj *Object) Ptr() unsafe.Pointer {
	return unsafe.Pointer(obj.obj)
}

func (obj *Object) call(args ...any) (*Object, error) {
	pArgs := C.PyTuple_New(C.long(len(args)))
	defer C.Py_DecRef(pArgs)
	for i, arg := range args {
		pArg, err := toCPyObject(arg)
		if err != nil {
			return nil, fmt.Errorf("cannot convert argument %d: %v", i, err)
		}
		C.PyTuple_SetItem(pArgs, C.long(i), pArg)
	}

	pValue := C.PyObject_CallObject(obj.obj, pArgs)
	if err := getPythonError(); err != nil {
		return nil, err
	}
	if pValue == nil || pValue == C.Py_None {
		return nil, nil
	}
	return newObject(pValue), nil
}

func (obj *Object) callS(funcName string, args ...any) (*Object, error) {
	fc, err := obj.attr(funcName)
	if err != nil {
		return nil, err
	}
	return fc.call(args...)
}

// CallS calls a function given its name and some arguments.
func (obj *Object) CallS(funcName string, args ...any) (*Object, error) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	return obj.callS(funcName, args...)
}

// Call calls this reference with a set of arguments and returns Python arguments
// casted to Go types.
func (obj *Object) Call(args ...any) (*Object, error) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	return obj.call(args...)
}

// Attr returns a reference to an attribute of a Python object.
func (obj *Object) Attr(name string) (*Object, error) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	return obj.attr(name)
}

func (obj *Object) attr(name string) (*Object, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	pObj := C.PyObject_GetAttrString(obj.obj, cName)
	if pObj == nil {
		return nil, getPythonError()
	}
	return newObject(pObj), nil
}

// String representation of the object.
func (obj *Object) String() string {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	sObj, err := obj.callS("__str__")
	if err != nil {
		return fmt.Sprintf("%T<%s>", obj, err.Error())
	}
	return cPyObjectToStr(sObj.obj)
}

func (obj *Object) goString() (string, error) {
	if C.PyStringCheck(obj.obj) == 0 {
		return "", newTypeError("cannot cast Python type %q to string", obj.typeName())
	}
	cRes := C.PyString_AsString(obj.obj)
	if cRes == nil {
		return "", getPythonError()
	}
	return C.GoString(cRes), nil
}

// GoString casts the Python object to a Go string.
func (obj *Object) GoString() (string, error) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	return obj.goString()
}

// IsNone returns true if the object is None.
func (obj *Object) IsNone() bool {
	return obj == nil || obj.obj == nil || obj.obj == C.Py_None
}

func cPyObjectToInt(o *C.PyObject) (int, error) {
	if C.PyLongCheck(o) == 0 {
		return 0, newTypeError("cannot cast Python type %q to int", cPyTypeName(o))
	}
	val := C.PyLong_AsLong(o)
	if err := getPythonError(); err != nil {
		return 0, err
	}
	return int(val), nil
}

// GoInt casts the Python object to a Go int.
func (obj *Object) GoInt() (int, error) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	return cPyObjectToInt(obj.obj)
}

func iterNextObject(iter *C.PyObject) (*Object, bool, error) {
	item := C.PyIter_Next(iter)
	if item == nil {
		return nil, false, getPythonError()
	}
	return newObject(item), true, nil
}

// GoSlice casts a Python iterable to a Go slice of Objects.
func (obj *Object) GoSlice() ([]*Object, error) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	iter := C.PyObject_GetIter(obj.obj)
	if err := getPythonError(); err != nil {
		return nil, err
	}
	defer C.Py_DecRef(iter)
	var vals []*Object
	for {
		v, ok, err := iterNextObject(iter)
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		vals = append(vals, v)
	}
	return vals, nil
}

func cPyTypeName(o *C.PyObject) string {
	tp := C.PyType(o)
	return C.GoString(tp.tp_name)
}

func (obj *Object) typeName() string {
	return cPyTypeName(obj.obj)
}

func cPyObjectToStr(o *C.PyObject) string {
	pyS := C.PyObject_Str(o)
	if pyS == nil {
		return "<pyobject cannot be converted to string>"
	}
	defer C.Py_DecRef(pyS)
	cS := C.PyString_AsString(pyS)
	return C.GoString(cS)
}

func strToCPyObject(s string) *C.PyObject {
	cS := C.CString(s)
	defer C.free(unsafe.Pointer(cS))
	return C.PyUnicode_FromStringAndSize(cS, C.long(len(s)))
}

func toCPythonMap(o any) (*C.PyObject, error) {
	iter := reflect.ValueOf(o).MapRange()
	dict := C.PyDict_New()
	for iter.Next() {
		k := iter.Key().Interface()
		pyK, err := toCPyObject(k)
		if err != nil {
			return nil, err
		}
		v := iter.Value().Interface()
		pyV, err := toCPyObject(v)
		if err != nil {
			return nil, err
		}
		if C.PyDict_SetItem(dict, pyK, pyV) != 0 {
			return nil, getPythonError()
		}
	}
	return dict, nil
}

func toCPythonList(o any) (*C.PyObject, error) {
	v := reflect.ValueOf(o)
	n := v.Len()
	list := C.PyList_New(C.long(n))
	for i := range n {
		pyElem, err := toCPyObject(v.Index(i).Interface())
		if err != nil {
			return nil, err
		}
		if C.PyList_SetItem(list, C.long(i), pyElem) != 0 {
			return nil, getPythonError()
		}
	}
	return list, nil
}

func toCPyObject(o any) (*C.PyObject, error) {
	switch t := o.(type) {
	case *Object:
		C.Py_IncRef(t.obj)
		return t.obj, nil
	case *C.PyObject:
		C.Py_IncRef(t)
		return t, nil
	case bool:
		var ti int64
		if t {
			ti = 1
		}
		return C.PyBool_FromLong(C.long(ti)), nil
	case int:
		return C.PyLong_FromLong(C.long(t)), nil
	case int32:
		return C.PyLong_FromLong(C.long(t)), nil
	case int64:
		return C.PyLong_FromLongLong(C.longlong(t)), nil
	case uint32:
		return C.PyLong_FromUnsignedLongLong(C.ulonglong(t)), nil
	case uint64:
		return C.PyLong_FromUnsignedLongLong(C.ulonglong(t)), nil
	case bfloat16.BFloat16:
		return C.PyFloat_FromDouble(C.double(t.Float32())), nil
	case float32:
		return C.PyFloat_FromDouble(C.double(t)), nil
	case float64:
		return C.PyFloat_FromDouble(C.double(t)), nil
	case string:
		return strToCPyObject(t), nil
	case []uint8:
		if len(t) == 0 {
			return C.PyBytes_FromStringAndSize(nil, 0), nil
		}
		cArg := (*C.char)(&(t[0]))
		return C.PyBytes_FromStringAndSize(cArg, C.long(len(t))), nil
	}
	switch reflect.TypeOf(o).Kind() {
	case reflect.Map:
		return toCPythonMap(o)
	case reflect.Slice:
		return toCPythonList(o)
	}
	return nil, errors.Errorf("cannot convert type %T to a PyObject", o)
}

func formatTraceback(ptype, pvalue, ptraceback *C.PyObject) (string, error) {
	traceback, err := importModule("traceback")
	if err != nil {
		return "", err
	}
	fe, err := traceback.attr("format_exception")
	if err != nil {
		return "", err
	}
	res, err := fe.call(ptype, pvalue, ptraceback)
	if res == nil || err != nil {
		return "", err
	}
	res, err = newObject(strToCPyObject("\n")).callS("join", res)
	if err != nil {
		return "", err
	}
	return res.goString()
}
