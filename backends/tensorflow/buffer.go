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
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/pygx/internal/pyelts"
)

// Buffer implements compute.Buffer for TensorFlow.
type Buffer struct {
	backend   *Backend
	deviceNum compute.DeviceNum
	shape     shapes.Shape
	flat      any
}

var _ compute.Buffer = (*Buffer)(nil)

func newBuffer(backend *Backend, deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (*Buffer, error) {
	if !shape.Ok() {
		return nil, errors.Errorf("invalid shape %s", shape)
	}
	data := dtypes.MakeAnySlice(shape.DType, shape.Size())
	dtypes.CopyAnySlice(data, flat)
	return &Buffer{
		backend:   backend,
		deviceNum: deviceNum,
		shape:     shape.Clone(),
		flat:      data,
	}, nil
}

// Backend returns the compute Backend that owns and manages this buffer.
func (b *Buffer) Backend() compute.Backend {
	return b.backend
}

// Finalize allows the client to inform the backend that the buffer is no longer needed.
func (b *Buffer) Finalize() error {
	b.flat = nil
	return nil
}

// Shape returns the shape for the buffer.
func (b *Buffer) Shape() (shapes.Shape, error) {
	return b.shape.Clone(), nil
}

// DeviceNum returns the deviceNum for the buffer.
func (b *Buffer) DeviceNum() (compute.DeviceNum, error) {
	return b.deviceNum, nil
}

// ToFlatData transfers the flat values of a buffer to a Go flat slice.
func (b *Buffer) ToFlatData(flat any) error {
	if b.flat == nil {
		return errors.Errorf("buffer is finalized")
	}
	dtypes.CopyAnySlice(flat, b.flat)
	return nil
}

// Data returns a slice pointing to the buffer storage memory directly.
func (b *Buffer) Data() (flat any, err error) {
	if b.flat == nil {
		return nil, errors.Errorf("buffer is finalized")
	}
	return b.flat, nil
}

// CopyToDevice copies the buffer to the deviceNum.
func (b *Buffer) CopyToDevice(deviceNum compute.DeviceNum) (compute.Buffer, error) {
	return nil, errors.Errorf("not implemented")
}

// toPythonElement converts the data of the buffer into a TensorFlow tensor.
func (b *Buffer) toPythonElement(tp ir.Type) (*pyelts.Element, error) {
	if b.flat == nil {
		return nil, errors.Errorf("buffer is finalized")
	}
	dtype, err := tfDType(b.shape.DType)
	if err != nil {
		return nil, err
	}
	npDType, err := dtype.Attr("as_numpy_dtype")
	if err != nil {
		return nil, err
	}
	arr, err := np.Mod().CallS("asarray", b.flat, npDType)
	if err != nil {
		return nil, err
	}
	reshaped, err := arr.CallS("reshape", b.shape.Dimensions)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tf.Mod().CallS("convert_to_tensor", reshaped, dtype))
}
