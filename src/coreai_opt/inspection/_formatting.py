# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tree-based formatting for model operation summaries using rich."""

from __future__ import annotations

import os

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from .types import ModelSummary, ModuleInfo, OpInfo

_FRAMEWORK_PATH_MARKERS = ("torch/nn/modules/", "torch/nn/functional", "torch/_")

_LEGEND = "Legend:  ■ module_name (module_type)  ◆ op_name [op_type]"


def _source_for_op(op: OpInfo) -> tuple[str, str]:
    """Return the source file path and code context as separate strings."""
    if not op.source_frames:
        return "", ""
    for f in reversed(op.source_frames):
        if not any(marker in f.filename for marker in _FRAMEWORK_PATH_MARKERS):
            frame = f
            break
    else:
        return "", ""
    try:
        rel_path = os.path.relpath(frame.filename)
    except ValueError:
        rel_path = frame.filename
    return f"{rel_path}:{frame.lineno}", frame.code_context


def _styled_op_label(op: OpInfo) -> Text:
    """Build the styled multi-line label for an op leaf node."""
    label = Text()

    # Line 1: ◆ op_name [op_type]
    label.append("◆ ")
    label.append(op.op_name, style="bold cyan")
    op_type_str = op.op_type if op.op_type else "?"
    label.append(" [")
    label.append(op_type_str, style="yellow")
    label.append("]")

    # Line 2: op inputs
    if op.inputs:
        input_names = ", ".join(inp.op_name for inp in op.inputs)
        label.append(f"\n  op inputs:  {input_names}")

    # Line 3: op outputs
    if op.outputs:
        output_names = ", ".join(out.op_name for out in op.outputs)
        label.append(f"\n  op outputs: {output_names}")

    # Lines 4-5: source
    source_path, source_code = _source_for_op(op)
    if source_path:
        label.append(f"\n  filepath:  {source_path}", style="dim")
        if source_code:
            label.append(f"\n  code:      {source_code}", style="dim")

    return label


def _styled_module_label(module: ModuleInfo) -> Text:
    """Build the styled label for a module node."""
    label = Text()
    label.append("■ ")
    label.append(module.module_name, style="green")
    label.append(" (")
    label.append(module.module_type, style="magenta")
    label.append(")")
    if module.input_ops:
        input_names = ", ".join(op.op_name for op in module.input_ops)
        label.append(f"\n    module inputs:  {input_names}", style="dim")
    if module.output_ops:
        output_names = ", ".join(op.op_name for op in module.output_ops)
        label.append(f"\n    module outputs: {output_names}", style="dim")
    return label


def _render_tree(module: ModuleInfo, tree: Tree) -> None:
    """Recursively add module children and ops to a rich Tree."""
    for child in module.child_modules.values():
        child_label = _styled_module_label(child)
        child_branch = tree.add(child_label)
        _render_tree(child, child_branch)

    for op in module.ops:
        label = _styled_op_label(op)
        tree.add(label)


def format_model_summary(summary: ModelSummary, colorize: bool | None = None) -> str:
    """Format a :class:`ModelSummary` as a module-hierarchy tree string.

    Args:
        summary (ModelSummary): The operation summary to format.
        colorize (bool | None): Whether to include ANSI color codes in the
            output. ``None`` (default) auto-detects based on terminal
            capabilities and environment variables (``NO_COLOR``,
            ``FORCE_COLOR``). ``True`` forces color on, ``False`` forces
            color off.

    Returns:
        str: The formatted tree as a string.
    """
    if not summary.model.ops and not summary.model.child_modules:
        return "(no compressible operations found)"

    # Root label
    root_label = Text("(")
    root_label.append(summary.model.module_type, style="magenta")
    root_label.append(")")
    if summary.model.input_ops:
        input_names = ", ".join(op.op_name for op in summary.model.input_ops)
        root_label.append(f"\n    module inputs:  {input_names}", style="dim")
    if summary.model.output_ops:
        output_names = ", ".join(op.op_name for op in summary.model.output_ops)
        root_label.append(f"\n    module outputs: {output_names}", style="dim")

    tree = Tree(root_label)
    _render_tree(summary.model, tree)

    console_kwargs: dict[str, bool] = {"highlight": False}
    if colorize is True:
        console_kwargs["force_terminal"] = True
    elif colorize is False:
        console_kwargs["no_color"] = True

    console = Console(**console_kwargs)
    with console.capture() as capture:
        console.print(tree)

    output = capture.get().rstrip("\n")
    return f"{_LEGEND}\n\n{output}"
