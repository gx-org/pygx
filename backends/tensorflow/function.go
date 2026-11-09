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
	"slices"

	"github.com/pkg/errors"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	"github.com/gx-org/gx/internal/base/cast"
	"github.com/gx-org/pygx/internal/gopy"
	"github.com/gx-org/pygx/internal/pyelts"
)

// Function implements compute.Function for TensorFlow.
type Function struct {
	bld  *Builder
	name string
	notimplemented.Function
}

var _ compute.Function = (*Function)(nil)

func (b *Builder) newFunction(name string) *Function {
	f := &Function{bld: b, name: name}
	f.ErrFn = func(op compute.OpType) error {
		return errors.Errorf("%v not implemented for builder %T", op, b)
	}
	return f
}

// Name of the function.
func (f *Function) Name() string {
	return f.name
}

// Builder owning the function.
func (f *Function) Builder() compute.Builder {
	return f.bld
}

func tfDType(dtype dtypes.DType) (*gopy.Object, error) {
	var name string
	switch dtype {
	case dtypes.Bool:
		name = "bool"
	case dtypes.BFloat16:
		name = "bfloat16"
	case dtypes.Float32:
		name = "float32"
	case dtypes.Float64:
		name = "float64"
	case dtypes.Int32:
		name = "int32"
	case dtypes.Int64:
		name = "int64"
	case dtypes.Uint32:
		name = "uint32"
	case dtypes.Uint64:
		name = "uint64"
	default:
		return nil, errors.Errorf("GX TensorFlow dtype for %v not implemented", dtype)
	}
	return tf.Mod().Attr(name)
}

func tfDTypeFromFlat(flat any) (*gopy.Object, error) {
	switch flat.(type) {
	case []bool:
		return tfDType(dtypes.Bool)
	case []bfloat16.BFloat16:
		return tfDType(dtypes.BFloat16)
	case []float32:
		return tfDType(dtypes.Float32)
	case []float64:
		return tfDType(dtypes.Float64)
	case []int32:
		return tfDType(dtypes.Int32)
	case []int64:
		return tfDType(dtypes.Int64)
	case []uint32:
		return tfDType(dtypes.Uint32)
	case []uint64:
		return tfDType(dtypes.Uint64)
	default:
		return nil, errors.Errorf("GX TensorFlow constant for %T not implemented", flat)
	}
}

func unaryOp(op string, x compute.Value) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](x)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tfmath.Mod().CallS(op, xEl.Obj))
}

// Exp returns the Op that represents the output of the corresponding operation.
func (f *Function) Exp(x compute.Value) (compute.Value, error) {
	return unaryOp("exp", x)
}

// Constant is a new TensorFlow constant.
func (f *Function) Constant(flat any, dims ...int) (compute.Value, error) {
	dtype, err := tfDTypeFromFlat(flat)
	if err != nil {
		return nil, err
	}
	npDType, err := dtype.Attr("as_numpy_dtype")
	if err != nil {
		return nil, err
	}
	arr, err := np.Mod().CallS("asarray", flat, npDType)
	if err != nil {
		return nil, err
	}
	reshaped, err := arr.CallS("reshape", dims)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tf.Mod().CallS("constant", reshaped, dtype))
}

func binaryOp(attr string, x, y compute.Value) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](x)
	if err != nil {
		return nil, err
	}
	yEl, err := cast.To[*pyelts.Element](y)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(xEl.Obj.CallS(attr, yEl.Obj))
}

// Add returns the element-wise sum of the two values.
// Standard broadcasting rules apply (see documentation).
func (f *Function) Add(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryOp("__add__", lhs, rhs)
}

// Div returns the element-wise division of the two values.
// Standard broadcasting rules apply (see documentation).
func (f *Function) Div(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryOp("__truediv__", lhs, rhs)
}

// Mul returns the element-wise multiplication of the two values.
// Standard broadcasting rules apply (see documentation).
func (f *Function) Mul(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryOp("__mul__", lhs, rhs)
}

// Sub returns the element-wise subtraction of the two values.
// Standard broadcasting rules apply (see documentation).
func (f *Function) Sub(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryOp("__sub__", lhs, rhs)
}

// Abs returns the Op that represents the output of the corresponding operation.
func (f *Function) Abs(x compute.Value) (compute.Value, error) {
	return unaryOp("abs", x)
}

// Log1p returns the expression log(x+1).
func (f *Function) Log1p(x compute.Value) (compute.Value, error) {
	return unaryOp("log1p", x)
}

// Neg returns the element-wise negation of x.
func (f *Function) Neg(x compute.Value) (compute.Value, error) {
	return unaryOp("negative", x)
}

// Sqrt returns the Op that represents the output of the corresponding operation.
func (f *Function) Sqrt(x compute.Value) (compute.Value, error) {
	return unaryOp("sqrt", x)
}

func binaryMathOp(op string, x, y compute.Value) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](x)
	if err != nil {
		return nil, err
	}
	yEl, err := cast.To[*pyelts.Element](y)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tfmath.Mod().CallS(op, xEl.Obj, yEl.Obj))
}

// Max returns the element-wise highest value among the two.
func (f *Function) Max(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("maximum", lhs, rhs)
}

// Min returns the element-wise smallest value among the two.
func (f *Function) Min(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("minimum", lhs, rhs)
}

// Equal performs element-wise equality check, returns boolean results with the same dimensions as input.
func (f *Function) Equal(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("equal", lhs, rhs)
}

// EqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) EqualTotalOrder(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("equal", lhs, rhs)
}

// NotEqual performs element-wise inequality check, returns boolean results with the same dimensions as input.
func (f *Function) NotEqual(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("not_equal", lhs, rhs)
}

// NotEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) NotEqualTotalOrder(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("not_equal", lhs, rhs)
}

// LessThan performs element-wise comparison, returns boolean results with the same dimensions as input.
func (f *Function) LessThan(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("less", lhs, rhs)
}

// LessThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) LessThanTotalOrder(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("less", lhs, rhs)
}

// LessOrEqual performs element-wise comparison, returns boolean results with the same dimensions as input.
func (f *Function) LessOrEqual(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("less_equal", lhs, rhs)
}

// LessOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) LessOrEqualTotalOrder(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("less_equal", lhs, rhs)
}

// GreaterThan performs element-wise comparison, returns boolean results with the same dimensions as input.
func (f *Function) GreaterThan(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("greater", lhs, rhs)
}

// GreaterThanTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) GreaterThanTotalOrder(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("greater", lhs, rhs)
}

// GreaterOrEqual performs element-wise comparison, returns boolean results with the same dimensions as input.
func (f *Function) GreaterOrEqual(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("greater_equal", lhs, rhs)
}

// GreaterOrEqualTotalOrder returns the element-wise operation.
// Standard broadcasting rules apply (see documentation).
func (f *Function) GreaterOrEqualTotalOrder(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryMathOp("greater_equal", lhs, rhs)
}

// Gather returns the Op that gathers slices from operand at the indices specified by startIndices.
func (f *Function) Gather(operand, startIndices compute.Value, indexVectorAxis int, offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int, indicesAreSorted bool) (compute.Value, error) {
	opEl, err := cast.To[*pyelts.Element](operand)
	if err != nil {
		return nil, err
	}
	indicesEl, err := cast.To[*pyelts.Element](startIndices)
	if err != nil {
		return nil, err
	}
	k := len(startIndexMap)
	if k == 0 || k > len(sliceSizes) {
		return nil, errors.Errorf("Gather: unsupported startIndexMap length %d for operand rank %d", k, len(sliceSizes))
	}
	for i, axis := range startIndexMap {
		if axis != i {
			return nil, errors.Errorf("Gather: unsupported startIndexMap %v (want 0..%d)", startIndexMap, k-1)
		}
	}
	if !slices.Equal(collapsedSliceAxes, startIndexMap) {
		return nil, errors.Errorf("Gather: unsupported collapsedSliceAxes %v (want 0..%d)", collapsedSliceAxes, k-1)
	}
	for i := range k {
		if sliceSizes[i] != 1 {
			return nil, errors.Errorf("Gather: collapsed axis %d has sliceSize %d, want 1", i, sliceSizes[i])
		}
	}
	for i := k; i < len(sliceSizes); i++ {
		if sliceSizes[i] < 0 {
			return nil, errors.Errorf("Gather: unsupported sliceSize %d for axis %d", sliceSizes[i], i)
		}
	}
	if indexVectorAxis < 0 || len(offsetOutputAxes) != len(sliceSizes)-k {
		return nil, errors.Errorf("Gather: unsupported indexVectorAxis %d or offsetOutputAxes %v", indexVectorAxis, offsetOutputAxes)
	}
	for i, axis := range offsetOutputAxes {
		if axis != indexVectorAxis+i {
			return nil, errors.Errorf("Gather: unsupported offsetOutputAxes %v for indexVectorAxis %d", offsetOutputAxes, indexVectorAxis)
		}
	}
	indicesObj := indicesEl.Obj
	if indexVectorAxis == 0 && k == 1 {
		indicesObj, err = tf.Mod().CallS("reshape", indicesObj, []int{1})
		if err != nil {
			return nil, err
		}
	}
	return pyelts.NewE(tf.Mod().CallS("gather_nd", opEl.Obj, indicesObj))
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis.
func (f *Function) Iota(shape shapes.Shape, iotaAxis int) (compute.Value, error) {
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis (%d) must be in the range [0, %d)", iotaAxis, shape.Rank())
	}
	dtype, err := tfDType(shape.DType)
	if err != nil {
		return nil, err
	}
	x, err := tf.Mod().CallS("range", 0, shape.Dimensions[iotaAxis], 1, dtype)
	if err != nil {
		return nil, err
	}
	reshapeDims := make([]int, shape.Rank())
	for i := range reshapeDims {
		reshapeDims[i] = 1
	}
	reshapeDims[iotaAxis] = shape.Dimensions[iotaAxis]
	x, err = tf.Mod().CallS("reshape", x, reshapeDims)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tf.Mod().CallS("broadcast_to", x, shape.Dimensions))
}

// Concatenate operands on the given axis.
func (f *Function) Concatenate(axis int, operands ...compute.Value) (compute.Value, error) {
	values := make([]*gopy.Object, len(operands))
	for i, op := range operands {
		el, err := cast.To[*pyelts.Element](op)
		if err != nil {
			return nil, err
		}
		values[i] = el.Obj
	}
	return pyelts.NewE(tf.Mod().CallS("concat", values, axis))
}

// Reshape reshapes x to the new dimensions.
// Total size cannot change, it's just a "reinterpretation" of the same flat data.
// The dtype remains the same, see ConvertDType to actually convert the values.
func (f *Function) Reshape(x compute.Value, dimensions ...int) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](x)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tf.Mod().CallS("reshape", xEl.Obj, dimensions))
}

// ConvertDType of x to dtype.
func (f *Function) ConvertDType(x compute.Value, dtype dtypes.DType) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](x)
	if err != nil {
		return nil, err
	}
	tfType, err := tfDType(dtype)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tf.Mod().CallS("cast", xEl.Obj, tfType))
}

// RNGBitGenerator generates the given shape filled with random bits using the Philox algorithm.
func (f *Function) RNGBitGenerator(state compute.Value, shape shapes.Shape) (newState compute.Value, values compute.Value, err error) {
	if shape.IsDynamic() {
		return nil, nil, errors.Errorf("RNGBitGenerator does not support dynamic shape %s", shape)
	}
	stateEl, err := cast.To[*pyelts.Element](state)
	if err != nil {
		return nil, nil, err
	}
	dtype, err := tfDType(shape.DType)
	if err != nil {
		return nil, nil, err
	}
	alg, err := tfrandom.Mod().Attr("Algorithm")
	if err != nil {
		return nil, nil, err
	}
	philox, err := alg.Attr("PHILOX")
	if err != nil {
		return nil, nil, err
	}
	none, err := tf.Mod().Attr("newaxis")
	if err != nil {
		return nil, nil, err
	}
	seed, err := tf.Mod().CallS("slice", stateEl.Obj, []int{0}, []int{2})
	if err != nil {
		return nil, nil, err
	}
	valuesObj, err := tfrandom.Mod().CallS("stateless_uniform", shape.Dimensions, seed, none, none, dtype, none, philox)
	if err != nil {
		return nil, nil, err
	}
	inc, err := f.Constant([]uint64{0, 1, 0}, 3)
	if err != nil {
		return nil, nil, err
	}
	newState, err = f.Add(state, inc)
	if err != nil {
		return nil, nil, err
	}
	return newState, pyelts.New(valuesObj), nil
}

// BroadcastInDim broadcasts the operand to an output with the given shape.
func (f *Function) BroadcastInDim(operand compute.Value, outputShape shapes.Shape, broadcastAxes []int) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](operand)
	if err != nil {
		return nil, err
	}
	opShape, err := f.bld.OpShape(operand)
	if err != nil {
		return nil, err
	}
	dims := slices.Repeat([]int{1}, outputShape.Rank())
	for i, ax := range broadcastAxes {
		dims[ax] = opShape.Dimensions[i]
	}
	reshaped, err := tf.Mod().CallS("reshape", xEl.Obj, dims)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tf.Mod().CallS("broadcast_to", reshaped, outputShape.Dimensions))
}

// Bitcast performs an elementwise bitcast operation from a dtype to another dtype.
func (f *Function) Bitcast(operand compute.Value, targetDType dtypes.DType) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](operand)
	if err != nil {
		return nil, err
	}
	dtype, err := tfDType(targetDType)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tf.Mod().CallS("bitcast", xEl.Obj, dtype))
}

func unaryBitwiseOp(op string, x compute.Value) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](x)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tfbitwise.Mod().CallS(op, xEl.Obj))
}

func binaryBitwiseOp(op string, x, y compute.Value) (compute.Value, error) {
	xEl, err := cast.To[*pyelts.Element](x)
	if err != nil {
		return nil, err
	}
	yEl, err := cast.To[*pyelts.Element](y)
	if err != nil {
		return nil, err
	}
	return pyelts.NewE(tfbitwise.Mod().CallS(op, xEl.Obj, yEl.Obj))
}

// BitwiseAnd returns the element-wise bitwise AND operation.
func (f *Function) BitwiseAnd(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryBitwiseOp("bitwise_and", lhs, rhs)
}

// BitwiseNot returns the element-wise bitwise NOT operation.
func (f *Function) BitwiseNot(x compute.Value) (compute.Value, error) {
	return unaryBitwiseOp("invert", x)
}

// BitwiseOr returns the element-wise bitwise OR operation.
func (f *Function) BitwiseOr(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryBitwiseOp("bitwise_or", lhs, rhs)
}

// BitwiseXor returns the element-wise bitwise XOR operator.
func (f *Function) BitwiseXor(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryBitwiseOp("bitwise_xor", lhs, rhs)
}

// ShiftLeft n bits. It implicitly preserves the sign bit if there is no overflow. So ShiftLeft(-1, 1) = -2.
func (f *Function) ShiftLeft(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryBitwiseOp("left_shift", lhs, rhs)
}

// ShiftRightArithmetic shifts right by n bits, preserving the sign bit. So ShiftRight(-2, 1) = -1.
func (f *Function) ShiftRightArithmetic(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryBitwiseOp("right_shift", lhs, rhs)
}

// ShiftRightLogical shifts right by n bits, destroying the sign bit.
func (f *Function) ShiftRightLogical(lhs compute.Value, rhs compute.Value) (compute.Value, error) {
	return binaryBitwiseOp("right_shift", lhs, rhs)
}
