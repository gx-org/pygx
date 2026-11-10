// Package pyobjects provides functions to test Python<->GX interop.
package pyobjects

import (
	"embed"

	"github.com/gx-org/gx/build/importers/embedpkg"
	"github.com/gx-org/gx/build/importers"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/gx/interp/engine"
	"github.com/gx-org/gx/stdlib/builtin"
	"github.com/gx-org/pygx/internal/pyelts"
)

func init() {
	embedpkg.RegisterPackage("pyobjects", buildPkg)
}

//go:embed *.gx
var staticFS embed.FS

var pkgBuilder = builtin.PackageBuilder{
	FullPath: "pyobjects",
	Builders: []builtin.Builder{
		builtin.ParseSource("pyobjects.gx").Absolute(),
		builtin.ImplementBuiltin("setValue", setValue),
		builtin.ImplementBuiltin("callWithEmptyBytes", callWithEmptyBytes),
		builtin.ImplementBuiltin("callWithNilBytes", callWithNilBytes),
		builtin.ImplementBuiltin("callWithMap", callWithMap),
	},
}

func buildPkg(bld importers.Builder) (importers.Package, error) {
	return builtin.BuilderParam{
		Builder: bld,
		FS:      staticFS,
	}.Build(pkgBuilder)
}

func setValue(env *engine.Env, call *ir.FuncCallExpr, recv ir.Element, genArgs, args []ir.Element) (_ []ir.Element, err error) {
	pyel, err := pyelts.CastFrom(args[0])
	if err != nil {
		return nil, err
	}
	arg1, err := pyelts.ObjectFrom(env, args[1])
	if err != nil {
		return nil, err
	}
	_, err = pyel.Obj.CallS("set", arg1)
	return nil, err
}

func callWithEmptyBytes(env *engine.Env, call *ir.FuncCallExpr, recv ir.Element, genArgs, args []ir.Element) ([]ir.Element, error) {
	pyel, err := pyelts.CastFrom(args[0])
	if err != nil {
		return nil, err
	}
	_, err = pyel.Obj.Call([]uint8{})
	return nil, err
}

func callWithNilBytes(env *engine.Env, call *ir.FuncCallExpr, recv ir.Element, genArgs, args []ir.Element) ([]ir.Element, error) {
	pyel, err := pyelts.CastFrom(args[0])
	if err != nil {
		return nil, err
	}
	var b []uint8
	_, err = pyel.Obj.Call(b)
	return nil, err
}

func callWithMap(env *engine.Env, call *ir.FuncCallExpr, recv ir.Element, genArgs, args []ir.Element) (_ []ir.Element, err error) {
	fn, err := pyelts.CastFrom(args[0])
	if err != nil {
		return nil, err
	}
	k, err := pyelts.CastFrom(args[1])
	if err != nil {
		return nil, err
	}
	v, err := pyelts.CastFrom(args[2])
	if err != nil {
		return nil, err
	}
	_, err = fn.Obj.Call(map[any]any{k.Obj: v.Obj})
	if err != nil {
		_, _ = fn.Obj.Call([]any{v.Obj, map[any]any{k.Obj: v.Obj}})
		_, _ = fn.Obj.Call(map[any]any{v.Obj: map[any]any{k.Obj: v.Obj}})
		for range 20 {
			_, _ = fn.Obj.Call(map[any]any{v.Obj: v.Obj, k.Obj: v.Obj})
		}
	}
	return nil, err
}
