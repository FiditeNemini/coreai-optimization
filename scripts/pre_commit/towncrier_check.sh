#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Pre-commit hook: run towncrier check if any changelog.d/ files are present on
# the branch or staged.
#
# Checks two sets of files:
#   1. Files differing between origin/main and HEAD — catches bad changelog entries
#      already committed to the PR branch.
#   2. Files currently staged — catches bad changelog entries about to be committed.
#
# If no changelog.d/ files appear in either set, the check is skipped entirely.

set -euo pipefail

if {
    git diff origin/main...HEAD --name-only
    git diff --cached --name-only
} |
    grep -q "^changelog\.d/"; then
    towncrier check --staged
fi
