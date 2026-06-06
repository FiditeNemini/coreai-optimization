#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

set -euo pipefail

echo "Cleaning up build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf -- *.egg-info
rm -rf .pytest_cache
rm -rf .ruff_cache
rm -rf .mypy_cache
find . -type d -name .nox -exec rm -rf {} + 2>/dev/null || true
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name ".coverage.*" -delete 2>/dev/null || true
rm -rf docs/build
rm -rf docs/src/api/generated
rm -f pytest-results.xml
rm -f task_stdout.log
echo "Clean up complete!"
