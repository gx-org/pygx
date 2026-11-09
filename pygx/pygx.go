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

// Command pygx provides utilities for GX in Python.
package main

import (
	"os"

	"flag"
	"github.com/spf13/cobra"
	"github.com/gx-org/pygx/pygx/pybind"
)

var rootCmd = &cobra.Command{
	Use:           "pygx",
	Short:         "GX Python utilities",
	SilenceUsage:  true,
	SilenceErrors: false,
}

func init() {
	rootCmd.SetErrPrefix("pygx:")
	rootCmd.AddCommand(pybind.Cmd())
}

func main() {
	google.Init()
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
