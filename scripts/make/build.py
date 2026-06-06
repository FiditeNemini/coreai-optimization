# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Build the coreai-opt package.

Usage:
    build.py          Build a standard wheel from the current version
    build.py --dev    Build a dev wheel with a PEP 440 .dev version
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# build.py ships to OSS, where `external/` is the repo root, so it imports the
# helpers via the plain `scripts.*` namespace — that resolves to `external/scripts/*`
# internally (PEP 420 merging) and to the OSS-root `scripts/*` post-export. The
# internal-only scripts use the explicit `external.scripts.*` form instead.
from scripts._utils import find_repo_root as _find_repo_root
from scripts.release.release_utils import (
    get_dev_release_version,
    get_package_version,
    write_version,
)


def run_build() -> None:
    """Run ``uv build`` to produce the wheel and sdist."""
    print(f"Building package with python (version: {sys.version})...")
    subprocess.run(["uv", "build"], check=True)
    print("Build complete! Check dist/ directory")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the coreai-opt package.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Build a dev wheel with a PEP 440 .dev version",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.dev:
        run_build()
        return
    repo_root = _find_repo_root(Path(__file__))
    original_version = get_package_version(repo_root)
    dev_version = os.environ.get("DEV_VERSION") or get_dev_release_version(original_version)
    try:
        write_version(repo_root, dev_version)
        print(f"Dev version: {dev_version}")
        run_build()
    finally:
        write_version(repo_root, original_version)


if __name__ == "__main__":
    main()
