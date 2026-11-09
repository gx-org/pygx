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
	"github.com/gx-org/gx/build/importers"
	"github.com/gx-org/gx/build/ir"
	"github.com/gx-org/pygx/internal/gopy"
	"github.com/gx-org/pygx/internal/pyelts"
)

type pkg struct {
	mod *gopy.Module
	ir  *ir.Package
}

func newPackage(mod *gopy.Module) importers.Package {
	p := &pkg{
		mod: mod,
		ir:  &ir.Package{},
	}
	p.ir.Runtime = p.runtime
	return p
}

func (p *pkg) runtime() (ir.PackageElement, error) {
	return p, nil
}

func (p *pkg) Type() ir.Type {
	return ir.PackageType()
}

func (p *pkg) Package() *ir.Package {
	return p.ir
}

func (p *pkg) Find(name string) (ir.Element, error) {
	obj, err := p.mod.Attr(name)
	if err != nil {
		return nil, err
	}
	return pyelts.New(obj), nil
}

// IR returns the current package intermediate representation.
func (p *pkg) IR() *ir.Package {
	return p.ir
}
