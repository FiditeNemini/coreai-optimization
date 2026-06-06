# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Helpers for in-place patching of ``pyproject.toml`` dep specs.

Used to substitute or append PEP 508 dep specs without editing source by
hand. Three editing operations:

- :func:`alias_dependency` — substitute references to one package name
  (and optional version) with a new PEP 508 spec, anywhere in the file
  that uses the quoted-list syntax (``[project.optional-dependencies]``,
  ``[dependency-groups]``).
- :func:`append_project_dependencies` — append PEP 508 specs to the
  ``[project] dependencies`` array.
- :func:`rename_package` — substitute the top-level ``[project] name``.

Plus :func:`parse_alias`, :func:`add_patch_args`, and :func:`apply_patch_args`
so any CLI can register ``--pin-dep`` / ``--alias-dep`` flags consistently.

All operations preserve comments, formatting, and indentation. Stdlib-only,
so callers can patch a ``pyproject.toml`` before any venv exists.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


def alias_dependency(repo_root: Path, oss_name: str, new_spec: str) -> None:
    """Replace references to ``oss_name`` in ``repo_root/pyproject.toml``.

    Substitutes a quoted PEP 508 dep entry with ``new_spec``. Two TOML
    basic-string forms are recognized:

    - single-line: ``"<oss_name>"`` or ``"<oss_name>==<version>"``
    - triple-quoted (TOML's ``\"\"\"...\"\"\"`` form), which may span
      multiple source lines (e.g. when combining a version with PEP 508
      environment markers such as ``; sys_platform == 'linux'``)

    Trailing commas are preserved. Covers ``[project.optional-dependencies]``
    and ``[dependency-groups]`` since both use the same quoted-list syntax.

    Useful when a library is published under different names in different
    distribution contexts (e.g. ``mylib`` upstream vs.
    ``vendor-mylib`` on a private index) so the pyproject can target one
    flavor without source edits.

    Args:
        repo_root (Path): Directory containing the target ``pyproject.toml``.
        oss_name (str): The package name to substitute (e.g. ``"mylib"``).
        new_spec (str): The replacement PEP 508 spec (e.g.
            ``"vendor-mylib==1.2.3"``).

    Raises:
        RuntimeError: If no references to ``oss_name`` are found, indicating
            a typo or stale alias.

    Note:
        Only TOML basic strings (double-quoted) are matched. Literal strings
        (single-quoted ``'...'`` / ``'''...'''``) are skipped, since PEP 508
        env markers conventionally use single quotes inside a double-quoted
        spec and the OSS pyproject doesn't use the literal-string form.

    """
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    # Single-line form: "<oss_name>" or "<oss_name>==<version>" on one line.
    single_line = re.compile(
        rf'^(?P<indent>\s*)"{re.escape(oss_name)}(?:==[^"]*)?"(?P<comma>,?)\s*$',
        flags=re.MULTILINE,
    )
    # Triple-quoted form: """<oss_name>...<spec>...""" possibly spanning
    # multiple source lines. `(?:\\?\s)*` after the opening """ tolerates
    # TOML's `\<newline>` line-continuation and any plain whitespace before
    # the package name. The `(?![\w-])` lookahead prevents matching a
    # longer package name (e.g. searching for `coreai` won't match
    # `coreai-extra` since the next char `-` is excluded). `[^"]*` then
    # absorbs the rest of the spec (version, extras, env markers) up to
    # the closing """.
    multi_line = re.compile(
        rf'^(?P<indent>\s*)"""(?:\\?\s)*{re.escape(oss_name)}(?![\w-])[^"]*"""(?P<comma>,?)\s*$',
        flags=re.MULTILINE,
    )
    content, count_single = single_line.subn(
        lambda m: f'{m["indent"]}"{new_spec}"{m["comma"]}',
        content,
    )
    content, count_multi = multi_line.subn(
        lambda m: f'{m["indent"]}"{new_spec}"{m["comma"]}',
        content,
    )
    if count_single + count_multi == 0:
        msg = f"No references to `{oss_name}` found in {pyproject}"
        raise RuntimeError(msg)
    pyproject.write_text(content, encoding="utf-8", newline="\n")


def append_project_dependencies(repo_root: Path, deps: list[str]) -> None:
    """Append entries to the ``[project] dependencies`` array.

    Locates the ``dependencies = [`` line and inserts each entry right
    before the closing ``]``. Tolerates indented closing brackets and
    expands an inline empty ``dependencies = []`` to the multi-line form.

    Args:
        repo_root (Path): Directory containing the target ``pyproject.toml``.
        deps (list[str]): PEP 508 dependency strings to append.

    Raises:
        RuntimeError: If the ``dependencies = [...]`` block isn't found, or
            the start line contains a non-empty inline list (which would
            require TOML parsing to round-trip safely).

    """
    pyproject = repo_root / "pyproject.toml"
    lines = pyproject.read_text(encoding="utf-8").splitlines()
    appended = [f'    "{d}",' for d in deps]
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("dependencies = ["))
    except StopIteration:
        msg = f"Could not locate `dependencies = [...]` block in {pyproject}"
        raise RuntimeError(msg) from None
    if lines[start].rstrip() == "dependencies = []":
        lines[start : start + 1] = ["dependencies = [", *appended, "]"]
    elif "]" in lines[start]:
        msg = (
            f"Inline non-empty `dependencies = [...]` form in {pyproject} is not "
            "supported; expand to multi-line first"
        )
        raise RuntimeError(msg)
    else:
        try:
            insert_idx = next(
                i for i, line in enumerate(lines[start + 1 :], start + 1) if line.strip() == "]"
            )
        except StopIteration:
            msg = f"Could not locate closing `]` for `dependencies = [...]` in {pyproject}"
            raise RuntimeError(msg) from None
        lines[insert_idx:insert_idx] = appended
    pyproject.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def rename_package(repo_root: Path, old_name: str, new_name: str) -> None:
    """Rewrite the ``[project] name`` field in ``repo_root/pyproject.toml``.

    Args:
        repo_root (Path): Directory containing the target ``pyproject.toml``.
        old_name (str): Expected current value of the ``[project] name`` field.
        new_name (str): Replacement value for the ``[project] name`` field.

    Raises:
        RuntimeError: If the expected ``name = "{old_name}"`` line is not
            found (e.g., the pyproject was renamed upstream and the alias
            is stale).

    """
    pyproject = repo_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    updated, count = re.subn(
        rf'^name = "{re.escape(old_name)}"$',
        lambda _m: f'name = "{new_name}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        msg = f'Did not find `name = "{old_name}"` in {pyproject}'
        raise RuntimeError(msg)
    pyproject.write_text(updated, encoding="utf-8", newline="\n")


def parse_alias(arg: str) -> tuple[str, str]:
    """Split an alias argument into ``(oss_name, new_spec)``.

    Args:
        arg (str): String of the form ``"<oss_name>=<new_spec>"``. Split on
            the first ``=`` only, so PEP 508 ``==<version>`` in the new spec
            stays intact.

    Returns:
        tuple[str, str]: Pair of ``(oss_name, new_spec)``.

    Raises:
        ValueError: If ``arg`` lacks a ``=`` separator or has an empty name
            / spec component. argparse's ``type=`` callback wraps this into
            an ``ArgumentTypeError`` with a user-friendly message.

    """
    oss_name, sep, new_spec = arg.partition("=")
    if not sep or not oss_name or not new_spec:
        msg = f"Invalid alias '{arg}'; expected '<oss_name>=<new_spec>'"
        raise ValueError(msg)
    return oss_name, new_spec


def add_patch_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--pin-dep`` and ``--alias-dep`` argparse flags on ``parser``.

    Both are repeatable. ``--pin-dep`` appends a PEP 508 spec to the
    ``[project] dependencies`` array. ``--alias-dep`` substitutes references
    to an OSS-named dep with a new spec. The flags share the same wording,
    types, and ``dest`` names across CLI front-ends so multiple scripts
    expose a consistent interface.

    Args:
        parser (argparse.ArgumentParser): The parser to extend in place.

    """
    parser.add_argument(
        "--pin-dep",
        action="append",
        default=[],
        dest="pin_deps",
        metavar="DEP",
        help=(
            "Append a PEP 508 spec to [project] dependencies. Repeatable, "
            "e.g. --pin-dep mylib==1.2.3"
        ),
    )
    parser.add_argument(
        "--alias-dep",
        action="append",
        default=[],
        dest="alias_deps",
        type=parse_alias,
        metavar="OSS_NAME=NEW_SPEC",
        help=(
            "Substitute references to OSS_NAME with NEW_SPEC across "
            "[project.optional-dependencies] and [dependency-groups]. "
            "Repeatable, e.g. --alias-dep mylib=vendor-mylib==1.2.3"
        ),
    )


def apply_patch_args(target: Path, args: argparse.Namespace) -> None:
    """Apply patches declared by ``--alias-dep`` and ``--pin-dep`` flags.

    Calls :func:`alias_dependency` for each ``--alias-dep`` pair (in order),
    then :func:`append_project_dependencies` for the collected ``--pin-dep``
    list. Logs each application to stdout. No-op when both lists are empty.

    Args:
        target (Path): Directory containing the target ``pyproject.toml``.
        args (argparse.Namespace): Parsed args containing ``alias_deps`` and
            ``pin_deps`` attributes (typically populated by
            :func:`add_patch_args`).

    """
    for oss_name, new_spec in args.alias_deps:
        alias_dependency(target, oss_name, new_spec)
        print(f"Aliased: {oss_name} -> {new_spec}")
    if args.pin_deps:
        append_project_dependencies(target, args.pin_deps)
        print(f"Pinned:  {', '.join(args.pin_deps)}")
