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

// Package pyelts provides Python elements for the GX interpreter.
package pyelts

import (
	"fmt"
	"go/token"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gx-org/gx/api/hostio"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/cgx/handle"
	"github.com/gx-org/gx/internal/base/cast"
	"github.com/gx-org/gx/internal/togo"
	"github.com/gx-org/gx/interp/elements"
	"github.com/gx-org/gx/interp/engine"
	"github.com/gx-org/gx/interp/materialise"
	"github.com/gx-org/pygx/internal/gopy"
)

// Element is a GX interpreter holding a python object.
type Element struct {
	Obj *gopy.Object
}

var _ hostio.Value = (*Element)(nil)
var _ engine.Func = (*Element)(nil)
var _ togo.CompileValue = (*Element)(nil)
var _ engine.NumericalElement = (*Element)(nil)
var _ engine.EvalShaper = (*Element)(nil)
var _ engine.WithAxes = (*Element)(nil)
var _ materialise.ElementMaterialiser = (*Element)(nil)
var _ materialise.Node = (*Element)(nil)
var _ elements.Unpacker = (*Element)(nil)

var gxModule = gopy.StaticImport("google3.third_party.gxlang.pygx.cgx")

// New returns a new interpreter element from a Python object.
func New(obj *gopy.Object) *Element {
	return &Element{Obj: obj}
}

// NewE returns a new interpreter element from a Python object (and an error).
func NewE(obj *gopy.Object, err error) (*Element, error) {
	if err != nil {
		return nil, err
	}
	return &Element{Obj: obj}, nil
}

// PyObject returns the Python hold by the element.
func (el *Element) PyObject() *gopy.Object {
	return el.Obj
}

// Type of the value.
func (el *Element) Type() ir.Type {
	return ir.UnknownType()
}

// ToHost transfers the value to the host.
func (el *Element) ToHost() (hostio.Value, error) {
	return el, nil
}

// Finalize releases the Python object.
func (el *Element) Finalize() error {
	el.Obj.Finalize()
	el.Obj = nil
	return nil
}

// SourceString returns a string representation of the object.
func (el *Element) SourceString(from *ir.File) string {
	return "PyObject*"
}

var file = &ir.File{}

type funcImpl struct {
	el *Element
}

// Name of the builtin function.
func (i *funcImpl) Name() string {
	return "<python function>"
}

// BuildFuncType builds the type of a function given how it is called.
func (i *funcImpl) BuildFuncType(tpcmp ir.TypeEval, call *ir.FuncCallExpr) (*ir.FuncType, error) {
	return ir.UnknownFType(), nil
}

// Implementation of the function, provided by the backend.
func (i *funcImpl) Implementation() any {
	return i.el.Call
}

// FuncIR returns the IR of the element as a function.
func (el *Element) FuncIR() ir.Func {
	return &ir.FuncBuiltin{
		FFile: file,
		FType: ir.UnknownFType(),
		Impl:  &funcImpl{el: el},
	}
}

// Unpack the element for axis evaluation.
func (el *Element) Unpack(ev ir.Evaluator) (ir.Element, error) {
	return el, nil
}

// Recv returns the receiver of the function.
func (el *Element) Recv() *engine.Receiver {
	return nil
}

// GoValue of the underlying element.
func (el *Element) GoValue() (any, error) {
	return el.Obj, nil
}

func toObjects[T ir.Element](env *engine.Env, args []T) ([]any, error) {
	anys := make([]any, len(args))
	for i, arg := range args {
		var err error
		anys[i], err = ObjectFrom(env, ir.Element(arg))
		if err != nil {
			return nil, err
		}
	}
	return anys, nil
}

// Call the element as a function.
func (el *Element) Call(env *engine.Env, call *ir.FuncCallExpr, genArgs, args []ir.Element) (_ []ir.Element, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("%s: %w", call.SourceString(env.File()), err)
		}
	}()
	anys := make([]any, len(args))
	for i, arg := range args {
		var err error
		anys[i], err = ObjectFrom(env, arg)
		if err != nil {
			return nil, err
		}
	}
	_, err = el.Obj.Call(anys...)
	return nil, err
}

// UnaryOp applies a unary operator on x.
func (el *Element) UnaryOp(env *engine.Env, expr *ir.UnaryExpr) (engine.NumericalElement, error) {
	switch expr.Src.Op {
	case token.ADD:
		return el, nil
	case token.SUB:
		return el.call(env, "__neg__")
	case token.NOT:
		// Python does not support overloading the not operator.
		// The invert operator is used instead: it is the logical negation for arrays of booleans.
		return el.call(env, "__invert__")
	default:
		return nil, fmt.Errorf("unary operator %q for %T not implemented", expr.Src.Op, el)
	}
}

func (el *Element) call(env *engine.Env, name string, args ...engine.NumericalElement) (engine.NumericalElement, error) {
	anys, err := toObjects(env, args)
	if err != nil {
		return nil, err
	}
	obj, err := el.Obj.CallS(name, anys...)
	if err != nil {
		return nil, err
	}
	return &Element{Obj: obj}, nil
}

// binaryMethods maps GX binary operators to the Python methods implementing them.
var binaryMethods = map[token.Token]string{
	token.ADD:  "__add__",
	token.SUB:  "__sub__",
	token.MUL:  "__mul__",
	token.QUO:  "__truediv__",
	token.EQL:  "__eq__",
	token.NEQ:  "__ne__",
	token.LSS:  "__lt__",
	token.LEQ:  "__le__",
	token.GTR:  "__gt__",
	token.GEQ:  "__ge__",
	token.AND:  "__and__",
	token.OR:   "__or__",
	token.XOR:  "__xor__",
	token.LAND: "__and__",
	token.LOR:  "__or__",
}

// BinaryOp applies a binary operator to x and y.
func (el *Element) BinaryOp(env *engine.Env, expr *ir.BinaryExpr, y engine.NumericalElement) (engine.NumericalElement, error) {
	if method, ok := binaryMethods[expr.Src.Op]; ok {
		return el.call(env, method, y)
	}
	// The other operators are computed by the backend:
	// the Python remainder is floored (instead of truncated in GX)
	// and TensorFlow tensors do not support shift operators.
	switch expr.Src.Op {
	case token.REM:
		return el.backendBinaryOp(env, compute.Function.Rem, y)
	case token.SHL:
		return el.backendBinaryOp(env, compute.Function.ShiftLeft, y)
	case token.SHR:
		return el.shiftRight(env, y)
	}
	return nil, fmt.Errorf("binary operator %s for %T not implemented", expr.Src.Op, el)
}

// backendBinary is a binary operation computed by a backend function.
type backendBinary func(fn compute.Function, x, y compute.Value) (compute.Value, error)

// backendBinaryOp computes a binary operation with the backend function of the engine.
func (el *Element) backendBinaryOp(env *engine.Env, op backendBinary, y engine.NumericalElement) (engine.NumericalElement, error) {
	yNode, err := cast.To[materialise.Node](y)
	if err != nil {
		return nil, err
	}
	return fromBackend(op(env.Engine().Graph(), el, yNode.Value()))
}

// shiftRight shifts the bits of x to the right.
// As in Go, the shift is logical for unsigned integers and arithmetic otherwise.
func (el *Element) shiftRight(env *engine.Env, y engine.NumericalElement) (engine.NumericalElement, error) {
	shape, err := el.EvalShape()
	if err != nil {
		return nil, err
	}
	if shape.DType.IsUnsigned() {
		return el.backendBinaryOp(env, compute.Function.ShiftRightLogical, y)
	}
	return el.backendBinaryOp(env, compute.Function.ShiftRightArithmetic, y)
}

// fromBackend returns the numerical element of a value computed by the backend.
func fromBackend(val compute.Value, err error) (engine.NumericalElement, error) {
	if err != nil {
		return nil, err
	}
	return cast.To[engine.NumericalElement](val)
}

// Materialise returns the element with all its values from the graph.
func (el *Element) Materialise(mat materialise.Materialiser) (materialise.Node, error) {
	return el, nil
}

// Value of the element as represented known by the backend.
func (el *Element) Value() compute.Value {
	return el
}

// Cast an element into a given data type.
func (el *Element) Cast(env *engine.Env, expr ir.Expr, target ir.Type) (engine.NumericalElement, error) {
	return fromBackend(env.Engine().Graph().ConvertDType(el, target.Kind().DType()))
}

// Reshape an element.
func (el *Element) Reshape(env *engine.Env, expr ir.Expr, axisLengths []engine.NumericalElement) (engine.NumericalElement, error) {
	dims := make([]int, len(axisLengths))
	for i, axisLength := range axisLengths {
		var err error
		if dims[i], err = elements.IntFromElement(axisLength); err != nil {
			return nil, err
		}
	}
	return fromBackend(env.Engine().Graph().Reshape(el, dims...))
}

func toAxisLength(obj *gopy.Object) (int, error) {
	if obj.IsNone() {
		return shapes.DynamicDim, nil
	}
	return obj.GoInt()
}

// EvalShape returns the shape of the Python object given its dtype and shape attributes.
func (el *Element) EvalShape() (shapes.Shape, error) {
	pyDType, err := el.Obj.Attr("dtype")
	if err != nil {
		return shapes.Invalid(), err
	}
	pyDTypeName, err := pyDType.Attr("name")
	if err != nil {
		return shapes.Invalid(), err
	}
	dtypeName, err := pyDTypeName.GoString()
	if err != nil {
		return shapes.Invalid(), err
	}
	dtype, err := dtypes.DTypeString(dtypeName)
	if err != nil {
		return shapes.Invalid(), err
	}
	if dtype == dtypes.InvalidDType {
		return shapes.Invalid(), fmt.Errorf("invalid dtype %q", dtypeName)
	}
	pyShape, err := el.Obj.Attr("shape")
	if err != nil {
		return shapes.Invalid(), err
	}
	pyDims, err := pyShape.GoSlice()
	if err != nil {
		return shapes.Invalid(), err
	}
	dims := make([]int, len(pyDims))
	for i, pyDim := range pyDims {
		dims[i], err = toAxisLength(pyDim)
		if err != nil {
			return shapes.Invalid(), err
		}
	}
	return shapes.Shape{DType: dtype, Dimensions: dims}, nil
}

// Axes returns the axis lengths of the Python object as a slice of elements.
func (el *Element) Axes(ev ir.Evaluator) (ir.FixedSlice, error) {
	shape, err := el.EvalShape()
	if err != nil {
		return nil, err
	}
	axes := ir.NewRank(shape.Dimensions).Axes()
	axlens := make([]ir.Element, len(axes))
	for i, axis := range axes {
		if axlens[i], err = ev.EvalExpr(axis.AsExpr()); err != nil {
			return nil, err
		}
	}
	return elements.NewSlice(ir.IntSliceType(), axlens)
}

// ObjectFromValue returns a Python object from a GX value.
func ObjectFromValue(val hostio.Value) (*gopy.Object, error) {
	cgx := gxModule.Mod()
	cVal := handle.Wrap(val)
	pyVal, err := cgx.CallS("Value", uint64(cVal))
	if err != nil {
		handle.Release(cVal)
		return nil, err
	}
	return cgx.CallS("python_value_from_gx_value", gopy.None(), gopy.None(), pyVal)
}

// ObjectFrom returns a Python object from an element.
func ObjectFrom(env *engine.Env, el ir.Element) (any, error) {
	goVal, err := togo.Value(env.Registry().Processor.LastInputs(), el)
	if err != nil {
		return nil, err
	}
	val, ok := goVal.(hostio.Value)
	if !ok {
		return goVal, nil
	}
	return ObjectFromValue(val)
}
