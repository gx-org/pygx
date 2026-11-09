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

// Package pybind implements the command to generate Python bindings for a GX package.
package pybind

import (
	_ "embed"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"text/template"

	"github.com/spf13/cobra"
)

const (
	commandName        = "pybind"
	targetFolderFlag   = "target_folder"
	defaultBackendFlag = "default_backend"
)

const pathPrefix = "google3"

//go:embed python.py.tmpl
var pyFile string
var pyTemplate = template.Must(template.New("pyTMPL").Parse(pyFile))

// Cmd returns the pybind command.
func Cmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   commandName + " <gx_package>",
		Short: "Generate Python bindings for a GX package",
		Args:  cobra.ExactArgs(1),
		RunE:  run,
	}
	cmd.Flags().String(targetFolderFlag, "", "target folder location")
	cmd.Flags().String(defaultBackendFlag, "", "GX backend to set as default")
	return cmd
}

type packageData struct {
	PackagePath string

	Backend string
	Init    string
}

func (data packageData) generate(tmpl *template.Template, dir, name string) error {
	filepath := filepath.Join(dir, name)
	f, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer f.Close()
	return tmpl.Execute(f, data)
}

func pyImportPathFromblazeTarget(target string) (string, string) {
	if i := strings.LastIndex(target, ":"); i != -1 {
		target = target[:i] + "/" + target[i+1:]
	}
	split := strings.Split(target, "/")
	for split[0] == "" {
		split = split[1:]
	}
	dir := split[:len(split)-1]
	last := split[len(split)-1]
	return strings.Join(dir, "."), last
}

func buildData(cmd *cobra.Command, pkgDir, pkgName string) (packageData, error) {
	data := packageData{
		PackagePath: path.Join(pkgDir, pkgName),
	}
	backend, err := cmd.Flags().GetString(defaultBackendFlag)
	if err != nil {
		return data, err
	}
	if backend == "" {
		return data, nil
	}
	pyImportPath, moduleName := pyImportPathFromblazeTarget(backend)
	data.Backend = fmt.Sprintf("from %s.%s import %s as backend", pathPrefix, pyImportPath, moduleName)
	data.Init = "backend.set_as_default()"
	return data, nil
}

func run(cmd *cobra.Command, args []string) error {
	targetFolder, err := cmd.Flags().GetString(targetFolderFlag)
	if err != nil {
		return err
	}
	libPath := args[0]

	// Prepare the target folder.
	pkgDir, pkgName := filepath.Split(libPath)
	pkgDir = strings.TrimSuffix(pkgDir, "/")
	pkgDir = strings.TrimPrefix(pkgDir, pathPrefix+"/")
	targetDir := filepath.Join(targetFolder, pkgDir)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return err
	}
	data, err := buildData(cmd, pkgDir, pkgName)
	if err != nil {
		return err
	}
	return data.generate(pyTemplate, targetDir, pkgName+".py")
}
