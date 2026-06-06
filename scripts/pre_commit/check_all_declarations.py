#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ast
import os
import sys


def check_all_in_init(filepath):
    """
    Checks if a Python file (typically __init__.py) contains the __all__ variable.
    """
    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=filepath)

    has_all = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    has_all = True
                    break
        if has_all:
            break

    if not has_all:
        print(
            f"Error: {filepath} is missing the '__all__' variable. "
            "Please define it to control wildcard imports."
        )  #
        return 1
    return 0


def main():
    """
    Main function to recursively check all __init__.py files in the project.
    """
    exit_code = 0
    for dirpath, _, filenames in os.walk(sys.argv[1]):
        if "tests" in dirpath:
            continue
        for filename in filenames:
            if filename == "__init__.py":
                full_path = os.path.join(dirpath, filename)
                if check_all_in_init(full_path) != 0:
                    exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
