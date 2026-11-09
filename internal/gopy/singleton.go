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

var none = &Object{obj: C.Py_None}

// None returns a None Python object.
func None() *Object {
	return none
}

var pyFalse = &Object{obj: C.Py_False}

// False returns a False Python object.
func False() *Object {
	return pyFalse
}

var pyTrue = &Object{obj: C.Py_True}

// True returns a None Python object.
func True() *Object {
	return pyTrue
}
