#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Verify that every Makefile target appears in .PHONY and vice-versa.

Also checks that the .PHONY list is alphabetically sorted.
"""

import re
import sys
from pathlib import Path


def _parse_phony_targets(makefile_text: str) -> list[str]:
    """Extract target names from the .PHONY declaration."""
    phony_targets: list[str] = []
    for line in makefile_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".PHONY:"):
            names = stripped.removeprefix(".PHONY:").split()
            phony_targets.extend(names)
    return phony_targets


def _parse_defined_targets(makefile_text: str) -> list[str]:
    """Extract target names from rule definitions (lines like 'target:' or 'target: deps')."""
    targets: list[str] = []
    for line in makefile_text.splitlines():
        # Skip comments, variable assignments, and recipe lines
        if line.startswith(("\t", "#")):
            continue
        # Match target definitions: word characters, hyphens at start of line, followed by ':'
        # Exclude lines with := ?= += (variable assignments) and :: (double-colon rules)
        match = re.match(r"^([a-zA-Z_][\w-]*):\s*[^:=]", line)
        if not match:
            match = re.match(r"^([a-zA-Z_][\w-]*):\s*$", line)
        if match:
            target = match.group(1)
            # Skip built-in/internal targets
            if not target.startswith(".") and target not in ("SHELL", "MAKEFLAGS"):
                targets.append(target)
    return targets


def main() -> int:
    """Check that .PHONY and target definitions are consistent and sorted."""
    # GNU Make searches in this order: GNUmakefile, makefile, Makefile.
    # Match the pre-commit `files` pattern so the script checks whichever variant exists.
    candidates = [Path("GNUmakefile"), Path("makefile"), Path("Makefile")]
    makefile_path = next((p for p in candidates if p.exists()), None)
    if makefile_path is None:
        print("Makefile: no Makefile found (expected GNUmakefile, makefile, or Makefile)")
        return 1

    text = makefile_path.read_text()

    # Join backslash-continuation lines before parsing so multi-line
    # .PHONY declarations and target rules are handled correctly.
    text = re.sub(r"\\\s*\n", " ", text)

    phony_targets = _parse_phony_targets(text)
    defined_targets = _parse_defined_targets(text)

    # Check for targets defined but missing from .PHONY
    phony_set = set(phony_targets)
    errors = [
        f"Target '{t}' is defined but missing from .PHONY"
        for t in defined_targets
        if t not in phony_set
    ]

    # Check for .PHONY entries with no corresponding target definition
    defined_set = set(defined_targets)
    errors.extend(
        f"'{t}' is listed in .PHONY but has no target definition"
        for t in phony_targets
        if t not in defined_set
    )

    # Check alphabetical ordering
    if phony_targets != sorted(phony_targets):
        errors.append(".PHONY targets are not in alphabetical order")

    if errors:
        for error in errors:
            print(f"Makefile: {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
