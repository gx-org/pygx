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
	"fmt"
	"sync"
)

// Module is a python loaded via an import.
type Module struct {
	*Object
}

func importModule(module string) (obj *Module, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("cannot load module %q: %v", module, err)
		}
	}()
	pName := strToCPyObject(module)
	defer C.Py_DecRef(pName)
	pModule := C.PyImport_Import(pName)
	if err := getPythonError(); err != nil {
		return nil, err
	}
	return &Module{
		Object: newObject(pModule),
	}, nil
}

// Import a python module.
func Import(path string) (*Module, error) {
	gil := EnsureGILState()
	defer ReleaseGILState(gil)
	return importModule(path)
}

// StaticModule is a module imported only once.
// It supports concurrency.
type StaticModule struct {
	path string

	once sync.Once
	mod  *Module
}

// StaticImport is import done on-demand only once.
func StaticImport(path string) *StaticModule {
	return &StaticModule{path: path}
}

// Mod returns the module or an error if it cannot be loaded.
func (sm *StaticModule) Mod() *Module {
	sm.once.Do(func() {
		var err error
		sm.mod, err = Import(sm.path)
		if err != nil {
			panic(fmt.Sprintf("cannot load Python module %s: %v", sm.path, err))
		}
	})
	return sm.mod
}
