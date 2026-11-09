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

package importer

import (
	"embed"

	"github.com/pkg/errors"
	"github.com/gx-org/gx/build/importers"
	"github.com/gx-org/gx/build/ir"
)

//go:embed py.gx
var fs embed.FS

func (imp *Importer) buildPy(bld importers.Builder) (importers.Package, error) {
	if imp.pypkg != nil {
		return imp.pypkg, nil
	}
	pypkg, err := bld.BuildFiles("", "py", fs, []string{"py.gx"})
	if err != nil {
		return pypkg, err
	}
	imp.pypkg = pypkg
	objType := imp.pypkg.IR().Decls.TypeByName("Obj")
	if objType == nil {
		return nil, errors.Errorf("cannot find Obj type")
	}
	objType.Underlying = ir.TypeFromExpr(ir.NewIdent(ir.UnknownType()))
	return imp.pypkg, nil
}
