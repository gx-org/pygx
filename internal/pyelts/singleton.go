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

import (
	"sync"

	"github.com/gx-org/pygx/internal/gopy"
)

// StaticObj is a static reference to a Python object.
type StaticObj struct {
	create func() (*gopy.Object, error)
	once   sync.Once
	el     *Element
}

// Singleton returns a Python object singleton.
func Singleton(create func() (*gopy.Object, error)) *StaticObj {
	return &StaticObj{create: create}
}

// El returns the python object.
func (s *StaticObj) El() *Element {
	s.once.Do(func() {
		obj, err := s.create()
		if err != nil {
			panic(err)
		}
		s.el = New(obj)
	})
	return s.el
}
