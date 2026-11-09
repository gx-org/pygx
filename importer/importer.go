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

// Package importer imports Python package into Python.
package importer

import (
	"strings"

	"github.com/gx-org/gx/build/importers"
	"github.com/gx-org/pygx/internal/gopy"
)

// Importer imports Python modules as GX packages.
type Importer struct {
	pypkg importers.Package
}

var _ importers.Importer = (*Importer)(nil)

// New Python importer.
func New() *Importer {
	return &Importer{}
}

const prefix = "python/"

// Support checks if the importer supports the import path given its prefix.
func (imp *Importer) Support(pkgpath string) bool {
	if pkgpath == "py" {
		return true
	}
	return strings.HasPrefix(pkgpath, prefix)
}

// Import a path.
func (imp *Importer) Import(bld importers.Builder, pkgpath string) (importers.Package, error) {
	if pkgpath == "py" {
		return imp.buildPy(bld)
	}
	pkgpath = strings.TrimPrefix(pkgpath, prefix)
	modpath := strings.ReplaceAll(pkgpath, "/", ".")
	mod, err := gopy.Import(modpath)
	if err != nil {
		return nil, err
	}
	return newPackage(mod), nil
}
