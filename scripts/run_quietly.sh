#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#
# Run a command quietly. Suppress output on success, replay to stderr on failure.
# Set QUIET=0 to disable suppression and run normally.
#
# Usage:
#   Executed:  scripts/run_quietly <cmd> [args...]
#   Sourced:   source scripts/run_quietly  → provides run_quietly() function

run_quietly() {
    if [[ "${QUIET:-1}" == "0" ]]; then
        "$@"
        return $?
    fi
    local output rc
    output=$("$@" 2>&1)
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "$output" >&2
    fi
    return $rc
}

# When executed directly (not sourced), run the arguments as a command.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    run_quietly "$@"
fi
