# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Patch a ``pyproject.toml`` in place with PEP 508 dep substitutions / appends.

CLI entrypoint for the helpers in :mod:`_pyproject_utils`. Used by the
``make env*`` targets to apply ``PIN_DEPS`` / ``ALIAS_DEPS`` to the active
``pyproject.toml`` before ``uv sync`` runs, so the resulting venv has the
patched deps without requiring source edits. Useful for testing against
specific dev versions of a renamed-or-pinned dep.

Patches are persistent. Revert with ``git checkout pyproject.toml`` (and
``uv.lock`` if checked in) when done.

Stdlib-only — runnable before any venv setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The patch script runs before any venv exists, so we can't rely on
# PYTHONPATH being configured by the surrounding tooling. Adding our own
# directory makes the sibling _pyproject_utils.py module importable
# regardless of cwd or invocation context.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pyproject_utils import add_patch_args, apply_patch_args


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Patch a pyproject.toml in place with --pin-dep (append base deps) "
            "and --alias-dep (substitute existing dep refs). Useful for "
            "testing against pinned or renamed dependencies."
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path.cwd(),
        help="Directory containing pyproject.toml (default: cwd).",
    )
    add_patch_args(parser)
    return parser.parse_args()


def main() -> None:
    """Apply pin / alias edits to the target pyproject.toml."""
    args = parse_args()
    if not args.pin_deps and not args.alias_deps:
        return

    target = args.target.resolve()
    if not (target / "pyproject.toml").is_file():
        sys.exit(f"Error: no pyproject.toml found in {target}")
    apply_patch_args(target, args)


if __name__ == "__main__":
    main()
