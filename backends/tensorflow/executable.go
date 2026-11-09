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

import (
	"github.com/pkg/errors"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
)

// Executable implements compute.Executable for TensorFlow.
type Executable struct{}

var _ compute.Executable = (*Executable)(nil)

// Finalize immediately frees resources associated to the executable.
func (e *Executable) Finalize() {}

// Inputs returns the parameters' names and shapes, in order created by the Builder.Parameter calls.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return nil, nil
}

// Outputs returns the computation's output shapes, in the order given to the Builder.Compile call.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return nil
}

// Execute runs the compiled computation.
func (e *Executable) Execute(inputs []compute.Buffer, donate []bool, defaultDevice compute.DeviceNum) ([]compute.Buffer, error) {
	return nil, errors.Errorf("not implemented")
}
