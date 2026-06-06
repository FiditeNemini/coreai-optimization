# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Sphinx extension: convert mermaid YAML front matter to directive options.

Standard markdown mermaid blocks use YAML front matter for titles::

    ```mermaid
    ---
    title: "My Title"
    ---
    flowchart LR
        A --> B
    ```

The MyST + sphinxcontrib-mermaid pipeline adds leading whitespace to the
opening ``---`` in the rendered HTML, which prevents mermaid.js from
recognizing the front matter block. This extension converts them to MyST
directive syntax during the Sphinx ``source-read`` phase::

    ```{mermaid}
    :caption: My Title

    flowchart LR
        A --> B
    ```

Source ``.md`` files remain in standard markdown format — only the
in-memory content seen by Sphinx is transformed.

This extension and ``sphinxcontrib.mermaid`` operate at different stages of the
Sphinx pipeline (read → parse → write):

* This extension hooks the ``source-read`` event (read phase) and rewrites the
  raw text before any parsing happens.
* ``sphinxcontrib.mermaid`` registers the ``mermaid`` directive, which is
  consumed later when MyST parses the (already-transformed) text into a doctree.

The two extensions do not overlap on directives or events, so their order in
``conf.py``'s ``extensions`` list does not matter.
"""

import re
from pathlib import Path

from sphinx.application import Sphinx

_MERMAID_FRONTMATTER_RE = re.compile(
    r'```mermaid\n---\ntitle:\s*(?:"([^"\n]+?)"|([^\n]+?))\s*\n---\n'
)


def _mermaid_replacement(m: re.Match[str]) -> str:
    title = m.group(1) or m.group(2)
    return f"```{{mermaid}}\n:caption: {title}\n\n"


def _transform_mermaid_frontmatter(_app: Sphinx, _docname: str, source: list[str]) -> None:
    """Replace mermaid YAML front matter with MyST ``:caption:`` directive option."""
    source[0] = _MERMAID_FRONTMATTER_RE.sub(_mermaid_replacement, source[0])


def _strip_svg_white_background(app: Sphinx, exception: Exception | None) -> None:
    """Strip the hardcoded ``background-color: white`` from generated mermaid SVGs.

    Mermaid v11 hardcodes a white background as an inline style on the root
    ``<svg>`` element, ignoring ``mmdc --backgroundColor transparent`` (which
    only affects PNG/PDF output). Rewrite the inline style so the SVG blends
    with the page theme.
    """
    if exception is not None:
        return
    images_dir = Path(app.outdir) / "_images"
    if not images_dir.is_dir():
        return
    for svg in images_dir.glob("mermaid-*.svg"):
        text = svg.read_text(encoding="utf-8")
        new_text = text.replace("background-color: white;", "background-color: transparent;")
        if new_text != text:
            svg.write_text(new_text, encoding="utf-8")


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the source-read and build-finished event handlers."""
    app.connect("source-read", _transform_mermaid_frontmatter)
    app.connect("build-finished", _strip_svg_white_background)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
