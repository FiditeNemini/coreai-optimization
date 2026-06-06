#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Script to run tests with optional coverage
# Usage:
#   ./run_tests.sh                          # Run all tests
#   ./run_tests.sh --cov                    # Run tests with coverage
#   ./run_tests.sh --marker "not slow"      # Run fast tests only
#   ./run_tests.sh --marker "slow"          # Run slow tests only
#   ./run_tests.sh --junit                  # Generate JUnit XML report
#   ./run_tests.sh --tb short               # Set traceback style (default: short)
#   ./run_tests.sh --pytest /path/to/pytest # Use specific pytest executable
#   ./run_tests.sh --workers N              # Number of workers (default: auto, 0 for no parallel)
#   ./run_tests.sh [options] <path>         # Run tests in specific path

set -e

# Default options
PYTEST_EXECUTABLE="pytest"
COVERAGE=""
TEST_MARKER=""
JUNIT_XML=""
DURATIONS=""
TRACEBACK="--tb=auto"
# Empty default so pytest falls back to `testpaths` in pyproject.toml
# (currently `["tests", "external/tests"]` so internal runs both internal-only
# and OSS-bound tests). A `--path X` argument or trailing positional path
# overrides this.
TEST_PATH=""
NUM_WORKERS="auto"
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --pytest)
        PYTEST_EXECUTABLE="$2"
        shift 2
        ;;
    --cov)
        COVERAGE="--cov=coreai_opt --cov-report=term-missing --cov-report=xml"
        shift
        ;;
    --marker)
        TEST_MARKER="-m \"$2\""
        shift 2
        ;;
    --junit)
        mkdir -p test-results
        JUNIT_XML="--junitxml=test-results/pytest-results.xml"
        shift
        ;;
    --durations)
        DURATIONS="--durations=$2"
        shift 2
        ;;
    --tb)
        TRACEBACK="--tb=$2"
        shift 2
        ;;
    --path)
        TEST_PATH="$2"
        shift 2
        ;;
    --workers)
        NUM_WORKERS="$2"
        shift 2
        ;;
    -*)
        # Unknown option - pass directly to pytest
        EXTRA_ARGS+=("$1")
        shift
        ;;
    *)
        # Positional argument - treat as test path
        TEST_PATH="$1"
        shift
        ;;
    esac
done

# Build and run pytest command directly
PYTEST_CMD="$PYTEST_EXECUTABLE -n $NUM_WORKERS $COVERAGE $TEST_MARKER $JUNIT_XML $DURATIONS $TRACEBACK $TEST_PATH ${EXTRA_ARGS[*]}"

echo "Running: $PYTEST_CMD"
eval "$PYTEST_CMD"
