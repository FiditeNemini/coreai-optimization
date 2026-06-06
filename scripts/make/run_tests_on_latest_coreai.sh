#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Run tests inside the latest-CoreAI virtual environment.
#
# Usage:
#   ./run_tests_on_latest_coreai.sh --path tests/export/
#   ./run_tests_on_latest_coreai.sh --path tests/coreai_utils/ --marker "not slow"
#
# All arguments are forwarded to run_tests.sh (--path is required).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv_latest_coreai"

echo "Running tests with latest CoreAI..."

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
(cd "$REPO_ROOT" && uv run --active python scripts/make/log_versions.py)
"$SCRIPT_DIR/run_tests.sh" "$@"

echo "All tests passed!"
