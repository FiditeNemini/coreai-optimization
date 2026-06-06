#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Log package versions and Python executable information."""

import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES: dict[str, list[str]] = {
    "Torch": ["torch", "torchvision", "torchao"],
    "CoreAI": ["coreai-core", "coreai-torch"],
}


def _get_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("=== Python ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    for section, packages in PACKAGES.items():
        print(f"=== {section} ===")
        for pkg in packages:
            print(f"{pkg}: {_get_version(pkg)}")


if __name__ == "__main__":
    main()
