#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#
# Standalone CLI wrapper for the ensure_npm_package function in scripts/utils.sh.
# Use when you want to invoke the check without sourcing utils.sh.
#
# Usage: ensure_npm_package.sh <npm_package> [binary_name] [required_registry]

set -euo pipefail

# shellcheck source=utils.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/utils.sh"

ensure_npm_package "$@"
