# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Repository root finding utilities.

This module provides a centralized way to find the repository root,
eliminating duplication across Python and bash scripts.
"""

from __future__ import annotations

from pathlib import Path

from pyreporoot import project_root

# Single source of truth for repo root marker
REPO_ROOT_MARKER = "pyproject.toml"


def find_repo_root(start_path: str | Path | None = None) -> Path:
    """Find repository root by locating pyproject.toml.

    Args:
        start_path: Starting path for search (defaults to current directory)

    Returns:
        Path to repository root

    Raises:
        FileNotFoundError: If repository root not found
    """
    if start_path is None:
        search_path = Path.cwd()
    else:
        search_path = Path(start_path)

    try:
        return Path(project_root(str(search_path.resolve()), root_files=[REPO_ROOT_MARKER]))
    except (FileNotFoundError, RuntimeError) as e:
        msg = f"Could not find repository root ({REPO_ROOT_MARKER} not found in parent directories)"
        raise FileNotFoundError(msg) from e
