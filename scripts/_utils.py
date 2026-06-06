# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Cross-cutting utilities for scripts/ tooling."""

from __future__ import annotations

import functools
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from start until a directory containing ``.git/`` is found.

    Note: this duplicates ``coreai_opt._utils.repo_utils.find_repo_root``
    intentionally — scripts under ``scripts/`` must work *before*
    ``coreai_opt`` is installed (e.g., during ``make env`` itself), so
    importing the canonical helper isn't possible. The marker differs
    (``.git/`` here vs. ``pyproject.toml`` there) because this lookup runs
    in scratch trees where ``pyproject.toml`` may exist outside the repo
    root (e.g., in nested test fixtures).

    Args:
        start (Path | None): Path to start searching from. Defaults to the
            directory containing this module.

    Returns:
        Path: Absolute path to the discovered repo root.

    Raises:
        RuntimeError: If no ``.git/`` directory is found in the ancestor chain.
    """
    base = (start or Path(__file__)).resolve()
    if base.is_file():
        base = base.parent
    for candidate in [base, *base.parents]:
        if (candidate / ".git").exists():
            return candidate
    msg = f"No .git/ directory found above {base}"
    raise RuntimeError(msg)


@functools.cache
def _default_scratch_dir() -> Path:
    """Return ``<repo_root>/.local/``."""
    return find_repo_root() / ".local"


@contextmanager
def scratch_file(*, suffix: str = "", parent_dir: Path | None = None) -> Iterator[Path]:
    """Yield a path to an empty temp file under parent_dir; remove it on exit.

    Args:
        suffix (str): Filename suffix including any leading dot, e.g., ".toml".
            Required when downstream tooling filters by extension (pre-commit's
            ``types:``, mime detection, etc.).
        parent_dir (Path | None): Directory to create the file in. Defaults to
            ``<repo_root>/.local/`` when ``None``; the repo root is resolved
            lazily on first use.

    Yields:
        Path: Absolute path to the empty file. The file is removed when the
            context exits, regardless of how it exits.
    """
    if parent_dir is None:
        parent_dir = _default_scratch_dir()
    parent_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=suffix, dir=parent_dir)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def scratch_dir(*, prefix: str | None = None, parent_dir: Path | None = None) -> Iterator[Path]:
    """Yield a path to a fresh temp directory under parent_dir; remove it (and contents) on exit.

    Args:
        prefix (str | None): Directory-name prefix. Defaults to tempfile's
            ``"tmp"`` prefix when ``None``.
        parent_dir (Path | None): Parent directory under which to create the
            temp directory. Defaults to ``<repo_root>/.local/`` when ``None``;
            the repo root is resolved lazily on first use.

    Yields:
        Path: Absolute path to the fresh temp directory. The directory and any
            contents are removed when the context exits.
    """
    if parent_dir is None:
        parent_dir = _default_scratch_dir()
    parent_dir.mkdir(parents=True, exist_ok=True)
    name = tempfile.mkdtemp(prefix=prefix, dir=parent_dir)
    path = Path(name)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
