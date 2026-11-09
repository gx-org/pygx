// Package pygxtestdeps imports GX packages to test with Python.
package pygxtestdeps

// #include "third_party/gxlang/gx/golang/binder/cgx/cgx.h"
import "C"

import (
	"github.com/gx-org/gx/api"
	"github.com/gx-org/gx/build/builder"
	"github.com/gx-org/gx/build/importers/embedpkg"
	"github.com/gx-org/gx/cgx/handle"
	"github.com/gx-org/gx/golang/backend"
	"github.com/gx-org/gx/stdlib"
	"github.com/gx-org/pygx/importer"

	// Packages from GX testing.
	_ "github.com/gx-org/gx/tests/testing"
	// Packages statically loaded for tests.
	_ "google3/third_party/gxlang/pygx/tests/pyimport/pyimport_gx"
	_ "github.com/gx-org/pygx/tests/pyobjects"
	_ "google3/third_party/gxlang/pygx/tests/structs/structs_gx"
	_ "google3/third_party/gxlang/pygx/tests/trace/trace_gx"
)

func newBuilder() *builder.Builder {
	return builder.New(
		stdlib.Importer(),
		embedpkg.New(),
		importer.New(),
	)
}

//export pygxtestdeps_graph_runtime
func pygxtestdeps_graph_runtime() C.struct_cgx_runtime_new_result {
	rtm, err := backend.New(newBuilder())
	if err != nil {
		return C.struct_cgx_runtime_new_result{
			error: (C.cgx_error)(handle.Wrap[error](err)),
		}
	}
	return C.struct_cgx_runtime_new_result{
		runtime: C.cgx_runtime(handle.Wrap[*api.Runtime](rtm)),
	}
}
