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

// Package tensorflow implements a TensorFlow backend for GX.
package tensorflow

import (
	"fmt"

	"github.com/pkg/errors"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gx-org/gx/api"
	"github.com/gx-org/gx/api/hostio"
	"github.com/gx-org/gx/build/builder"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/eagereval"
	"github.com/gx-org/pygx/internal/gopy"
	"github.com/gx-org/pygx/internal/pyelts"
)

// Backend implements compute.Backend for TensorFlow.
type Backend struct{}

var _ compute.Backend = (*Backend)(nil)
var _ eagereval.EagerBackend = (*Backend)(nil)

var capabilities = compute.Capabilities{}

var np = gopy.StaticImport("numpy")
var tf = gopy.StaticImport("tensorflow")
var tfbitwise = gopy.StaticImport("tensorflow.bitwise")
var tfmath = gopy.StaticImport("tensorflow.math")
var tfrandom = gopy.StaticImport("tensorflow.random")

// New returns a new TensorFlow backend.
func New(bld *builder.Builder) *api.Runtime {
	return api.NewRuntime(&Backend{}, bld)
}

// Name returns the short name of the backend.
func (b *Backend) Name() string {
	return "tensorflow"
}

// ToElement returns the interpreter element representing a value computed by TensorFlow.
// TensorFlow values are Python objects which already are numerical elements.
func (b *Backend) ToElement(val compute.Value, tp ir.Type) (eagereval.NumericalNode, error) {
	switch valT := val.(type) {
	case *pyelts.Element:
		return valT, nil
	case *hostio.DeviceArray:
		return b.ToElement(valT.Buffer(), tp)
	case *Buffer:
		return valT.toPythonElement(tp)
	default:
		return nil, errors.Errorf("cannot convert %T to a python element", val)
	}
}

// String returns the same as Name.
func (b *Backend) String() string {
	return b.Name()
}

// Description is a longer description of the Backend that can be used to pretty-print.
func (b *Backend) Description() string {
	return "TensorFlow Backend"
}

// NumDevices returns the number of devices available for this Backend.
func (b *Backend) NumDevices() int {
	return 1
}

// DeviceDescription returns a description of the device at the given deviceNum.
func (b *Backend) DeviceDescription(deviceNum compute.DeviceNum) string {
	return fmt.Sprintf("TensorFlow device #%d", deviceNum)
}

// Capabilities returns information about what is supported by this backend.
func (b *Backend) Capabilities() compute.Capabilities {
	return capabilities
}

// Builder creates a new builder used to define a newly named computation.
func (b *Backend) Builder(name string) compute.Builder {
	return &Builder{name: name}
}

// BufferFromFlatData transfers data from Go given as a flat slice to the deviceNum, and returns the corresponding Buffer.
func (b *Backend) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	return newBuffer(b, deviceNum, flat, shape)
}

// HasSharedBuffers returns whether the backend supports shared buffers.
func (b *Backend) HasSharedBuffers() bool {
	return false
}

// NewSharedBuffer returns a shared buffer that can be both used as input for execution of computations and directly read or mutated by the clients.
func (b *Backend) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (buffer compute.Buffer, flat any, err error) {
	return nil, nil, errors.Errorf("not implemented")
}

// Finalize releases all the associated resources immediately and makes the backend invalid.
func (b *Backend) Finalize() {}

// IsFinalized returns true if the backend is finalized.
func (b *Backend) IsFinalized() bool {
	return false
}
