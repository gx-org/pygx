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
	"github.com/gx-org/pygx/internal/pyelts"
)

// Builder implements compute.Builder for TensorFlow.
type Builder struct {
	name string
}

var _ compute.Builder = (*Builder)(nil)

// Name returns the name of the computation being built.
func (b *Builder) Name() string {
	return b.name
}

// Main returns the main function of this computation.
func (b *Builder) Main() compute.Function {
	return b.newFunction(compute.MainName)
}

// NewFunction creates a new named function within this builder.
func (b *Builder) NewFunction(name string) (compute.Function, error) {
	return nil, errors.Errorf("not implemented")
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op compute.Value) (shapes.Shape, error) {
	obj, err := pyelts.CastFrom(op)
	if err != nil {
		return shapes.Invalid(), err
	}
	return obj.EvalShape()
}

// DistributedSPMD creates a computation that will be executed on multiple devices in SPMD fashion.
func (b *Builder) DistributedSPMD(numDevices int) error {
	return errors.Errorf("not implemented")
}

// DistributedAutoSharding creates a computation that will be executed on multiple devices with auto-sharding.
func (b *Builder) DistributedAutoSharding(meshes ...compute.Mesh) error {
	return errors.Errorf("not implemented")
}

// DeviceAssignment assigns the concrete devices to the computation.
func (b *Builder) DeviceAssignment(devices ...compute.DeviceNum) error {
	return errors.Errorf("not implemented")
}

// Compile compiles the computation built.
func (b *Builder) Compile() (compute.Executable, error) {
	return nil, errors.Errorf("not implemented")
}
