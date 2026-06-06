#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Insert or refresh a license header on source files.

The header is rendered from a ``--license-file`` Jinja template whose ``year``
variable is filled with the file's copyright years -- drawn from git history
plus the current year, and floored by ``--start-year``. Comment syntax is picked
from the file's extension: ``#`` for Python/shell/Makefile, ``//`` for
JavaScript, ``/* ... */`` for CSS, and ``{#- ... -#}`` for Jinja templates.
Because the header is recognised by the template's fixed license text, any
license works: a matching header is refreshed in place (rewritten only when it
changes), while a leading comment resembling a different license fails the hook
for manual resolution instead of stacking a second header.

Years come from ``git log``, so run with full history; a shallow clone yields
incomplete headers.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

# Substring marking a template line as containing a Jinja expression. Lines
# carrying an expression vary by render and are excluded from the invariant
# text used to recognise our header.
_JINJA_EXPRESSION_MARKER = "{{"
# A run of three or more consecutive years (span >= 2) collapses to "first-last".
_MIN_COLLAPSIBLE_SPAN = 2
# Marks a leading comment block as a license header (a copyright year or SPDX
# tag). When such a block is present but does not match the template's invariant
# lines, the hook fails rather than stacking a second header on top of it.
_LICENSE_SIGNAL = re.compile(r"copyright\s+\d{4}|spdx", re.IGNORECASE)


@dataclass(frozen=True)
class CommentStyle:
    """Comment syntax used to render and recognise a leading license block.

    Line styles (e.g. ``# foo``) leave ``block_open``/``block_close`` empty and
    detect a header by a contiguous run of lines beginning with the stripped
    ``line_prefix``. Block styles wrap the body between an opener and a closer;
    detection looks for the contiguous lines from the opener through the
    closer.

    Attributes:
        name (str): Human-readable identifier (used in error messages).
        line_prefix (str): Prefix for non-blank text lines (e.g. ``"# "``).
        blank_marker (str): Rendering for a blank text line (e.g. ``"#"``).
        block_open (str): Block opener line, or ``""`` for a line-comment style.
        block_close (str): Block closer line, or ``""`` for a line-comment
            style.
    """

    name: str
    line_prefix: str
    blank_marker: str
    block_open: str = ""
    block_close: str = ""


_HASH = CommentStyle(name="hash", line_prefix="# ", blank_marker="#")
_SLASH = CommentStyle(name="slash", line_prefix="// ", blank_marker="//")
_BLOCK_C = CommentStyle(
    name="block-c", line_prefix=" * ", blank_marker=" *", block_open="/*", block_close=" */"
)
# ``.html``/``.htm`` files in this codebase are Jinja templates; we use Jinja's
# own ``{#- ... -#}`` block-comment syntax so the license is invisible in the
# rendered HTML output. The name is anchored on the file type rather than on
# the comment-syntax language to avoid colliding with Jinja the templating
# engine (which renders the license template itself, see ``main``).
_BLOCK_HTML = CommentStyle(
    name="block-html", line_prefix="  ", blank_marker="", block_open="{#-", block_close="-#}"
)

# Map file extension -> comment style. Files whose extension is missing fall
# back to ``_HASH``, which covers Makefile, Dockerfile, and similar.
_STYLE_BY_SUFFIX: dict[str, CommentStyle] = {
    ".js": _SLASH,
    ".css": _BLOCK_C,
    ".html": _BLOCK_HTML,
    ".htm": _BLOCK_HTML,
}


class HeaderConflictError(Exception):
    """Raised when a file already has a license-like header that is not ours."""


def normalize(years: set[int]) -> str:
    """Render a set of years in compact revised-work notation.

    A run of three or more consecutive years becomes ``"first-last"``; runs of
    one or two years are listed individually. Pieces are joined with ``", "``.

    Args:
        years (set[int]): Years in which the file was revised.

    Returns:
        str: The compact year string, e.g. ``"2016-2018, 2020, 2026"``.

    """
    ordered = sorted(years)
    if not ordered:
        return ""
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for year in ordered[1:]:
        if year == previous + 1:
            previous = year
        else:
            runs.append((start, previous))
            start = previous = year
    runs.append((start, previous))

    pieces: list[str] = []
    for first, last in runs:
        if last - first >= _MIN_COLLAPSIBLE_SPAN:
            pieces.append(f"{first}-{last}")
        else:
            pieces.extend(str(year) for year in range(first, last + 1))
    return ", ".join(pieces)


def render_header_block(
    template: Template, *, year_string: str, style: CommentStyle = _HASH
) -> list[str]:
    """Render the comment-prefixed header lines from a parsed Jinja template.

    Renders the template with ``year`` set to ``year_string``, strips leading
    and trailing blank lines (blank lines between text are preserved), and
    wraps every line in the chosen comment style.

    Args:
        template (Template): Pre-parsed Jinja template (hoist parsing out of
            per-file loops; one parse per ``main`` invocation is enough).
        year_string (str): Pre-computed copyright year string.
        style (CommentStyle): Comment syntax to use; defaults to ``#``.

    Returns:
        list[str]: The header lines.

    """
    rendered = template.render(year=year_string)
    lines = rendered.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    body = [f"{style.line_prefix}{line}" if line.strip() else style.blank_marker for line in lines]
    if style.block_open:
        return [style.block_open, *body, style.block_close]
    return body


def build_new_content(
    original: str,
    block_lines: list[str],
    invariant_lines: list[str],
    style: CommentStyle = _HASH,
) -> str:
    """Return ``original`` with the license header inserted or refreshed.

    On a refresh ("ours"), the existing header's comment lines are replaced in
    place; everything around them — including the blank lines the user chose
    between the header and the body — is left untouched. On a fresh insert
    ("absent"), the header is followed by two blank lines (PEP 8 spacing for a
    top-level class or function definition).

    Args:
        original (str): Current file contents.
        block_lines (list[str]): Header lines from ``render_header_block``.
        invariant_lines (list[str]): Template lines that identify our header.
        style (CommentStyle): Comment syntax used for both rendering and
            recognition.

    Returns:
        str: Updated contents, always ending with a single newline.

    Raises:
        HeaderConflictError: If the file already carries a license-like leading
            comment block that does not match the template.

    """
    lines = original.splitlines()
    preamble: list[str] = []
    if lines and lines[0].startswith("#!"):
        preamble = [lines[0]]
        lines = lines[1:]

    status, body = _split_header(lines, invariant_lines, style)
    if status == "conflict":
        msg = "existing license-like header does not match the template"
        raise HeaderConflictError(msg)

    spaced_body = body if status == "ours" else (["", "", *body] if body else [])
    assembled = (
        [*preamble, "", *block_lines, *spaced_body] if preamble else [*block_lines, *spaced_body]
    )
    while assembled and not assembled[-1].strip():
        assembled.pop()
    return "\n".join(assembled) + "\n"


def process_file(
    path: Path,
    template: Template,
    *,
    current_year: int,
    start_year: int | None,
    invariant_lines: list[str],
) -> bool:
    """Insert or refresh the header on ``path``; return whether action is needed.

    Args:
        path (Path): File to process.
        template (Template): Pre-parsed Jinja template for the license header.
        current_year (int): UTC year of this run, always included in the header.
        start_year (int | None): When set, git-history years before it are
            dropped; the current year is still kept.
        invariant_lines (list[str]): Template lines that identify our header.

    Returns:
        bool: ``True`` if the file was rewritten or has a conflicting header
        that needs manual resolution; ``False`` if it was already correct.

    """
    style = _pick_style(path)
    original = path.read_text(encoding="utf-8")
    years = _git_commit_years(path)
    if start_year is not None:
        years = {year for year in years if year >= start_year}
    years.add(current_year)
    block_lines = render_header_block(template, year_string=normalize(years), style=style)
    try:
        updated = build_new_content(original, block_lines, invariant_lines, style)
    except HeaderConflictError as error:
        msg = f"{path}: {error}; refusing to add a second header. Resolve it manually."
        print(msg, file=sys.stderr)
        return True
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8", newline="\n")
    print(f"Updated {path}")
    return True


def main(argv: list[str] | None = None) -> int:
    """Add or refresh the license header on each path passed as an argument.

    Returns:
        int: ``1`` if any file was rewritten or needs manual resolution, else ``0``.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license-file", required=True, type=Path)
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Ignore git-history years before this year; the current year is always kept.",
    )
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    template_text = args.license_file.read_text(encoding="utf-8")
    template = Template(template_text)
    invariant_lines = _template_invariant_lines(template_text)
    current_year = datetime.datetime.now(datetime.UTC).year

    needs_action = False
    for path in args.files:
        if process_file(
            path,
            template,
            current_year=current_year,
            start_year=args.start_year,
            invariant_lines=invariant_lines,
        ):
            needs_action = True
    return 1 if needs_action else 0


def _pick_style(path: Path) -> CommentStyle:
    """Return the comment style for ``path``'s extension, or ``#`` as fallback."""
    return _STYLE_BY_SUFFIX.get(path.suffix.lower(), _HASH)


def _template_invariant_lines(template_text: str) -> list[str]:
    """Return the template's expression-free, non-blank lines.

    These lines (the license text) are identical in every rendered header
    regardless of Jinja variables, so their presence in a leading comment block
    identifies the block as one of ours.

    Args:
        template_text (str): Raw Jinja template contents.

    Returns:
        list[str]: The constant template lines.

    """
    return [
        line
        for line in template_text.splitlines()
        if line.strip() and _JINJA_EXPRESSION_MARKER not in line
    ]


def _split_header(
    lines: list[str], invariant_lines: list[str], style: CommentStyle
) -> tuple[str, list[str]]:
    """Classify and split off the leading comment block.

    Drops leading blank lines, then inspects the leading comment block defined
    by ``style``. When the block is recognised as ours, the blank lines after
    it are returned as part of ``body`` so the caller can preserve the user's
    spacing; on a fresh insert the body's leading blanks have already been
    stripped above.

    Args:
        lines (list[str]): File lines with any shebang already removed.
        invariant_lines (list[str]): Template lines that identify our header.
        style (CommentStyle): Comment syntax to recognise.

    Returns:
        tuple[str, list[str]]: ``(status, body)`` where ``status`` is ``"ours"``
        (the leading block matched the template; ``body`` is everything after
        it, blanks intact), ``"conflict"`` (a license-like block that is not
        ours), or ``"absent"`` (no license-like leading block; ``body`` has had
        its leading blanks stripped).

    """
    body = list(lines)
    while body and not body[0].strip():
        body.pop(0)

    if style.block_open:
        if not body or not body[0].lstrip().startswith(style.block_open):
            return "absent", body
        close_token = style.block_close.strip()
        run_end = next(
            (i + 1 for i, line in enumerate(body) if line.strip().endswith(close_token)),
            None,
        )
        if run_end is None:
            return "absent", body
    else:
        line_marker = style.line_prefix.strip()
        run_end = 0
        while run_end < len(body) and body[run_end].startswith(line_marker):
            run_end += 1

    if run_end == 0:
        return "absent", body
    run_text = "\n".join(body[:run_end])
    if invariant_lines and all(line in run_text for line in invariant_lines):
        return "ours", body[run_end:]
    if _LICENSE_SIGNAL.search(run_text):
        return "conflict", body
    return "absent", body


def _git_commit_years(path: Path) -> set[int]:
    """Return the set of author-date years in which ``path`` was committed.

    Returns an empty set when the file is untracked, git is unavailable, or the
    path lies outside a repository.

    Args:
        path (Path): File whose git history to inspect.

    Returns:
        set[int]: Four-digit years drawn from the file's commit history.

    """
    try:
        result = subprocess.run(
            ["git", "log", "--follow", "--format=%ad", "--date=format:%Y", "--", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()
    return {int(token) for token in result.stdout.split() if token.isdigit()}


if __name__ == "__main__":
    sys.exit(main())
