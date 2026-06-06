# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph module op discovery implementation.

Walks an exported ``torch.fx.GraphModule`` graph and extracts operation
metadata from FX node attributes and metadata dictionaries.
"""

from __future__ import annotations

import re

import torch
from torch.fx import Node

from coreai_opt._utils.torch_utils import (
    get_node_type as _get_node_type,
    normalize_module_fqn as _normalize_module_fqn,
)
from coreai_opt.base_model_compressor import _BaseModelCompressor
from coreai_opt.quantization.config.quantization_config import ExecutionMode

from .types import (
    ModelSummary as _ModelOpSummary,
    ModuleContext as _ModuleContext,
    ModuleInfo as _ModuleSummary,
    OpInfo as _OpInfo,
    SourceFrame as _SourceFrame,
)

# The function name used to identify relevant source frames.
# Only frames from ``forward`` methods are kept; all other frames
# (framework dispatch, C++ internals, etc.) are discarded.
_FORWARD_FUNCTION_NAME = "forward"


def _extract_module_stack(node: Node) -> tuple[_ModuleContext, ...]:
    """Build the module nesting hierarchy from ``nn_module_stack`` metadata."""
    stack = node.meta.get("nn_module_stack", {})
    return tuple(
        _ModuleContext(module_name=_normalize_module_fqn(module_fqn), module_type=module_type)
        for module_fqn, module_type in stack.values()
    )


def _parse_stack_trace(stack_trace: str | None) -> tuple[_SourceFrame, ...]:
    """Parse the ``stack_trace`` metadata string into filtered source frames.

    The ``stack_trace`` stored in ``node.meta["stack_trace"]`` is a multi-line
    string formatted like a Python traceback::

        File "path/to/file.py", line 42, in forward
          x = self.conv(x)

    Only frames from ``forward()`` methods are kept, filtering out framework
    dispatch machinery, C++ internals, and other non-informative frames.
    """
    if not stack_trace:
        return ()

    frames: list[_SourceFrame] = []
    lines = stack_trace.strip().splitlines()
    # Lines come in pairs: the first is a location header of the form
    #   File "path/to/file.py", line 42, in forward
    # and the second is the source line at that location:
    #   x = self.conv(x)
    # We parse each header, peek ahead for the source line, and keep
    # only frames originating from ``forward()`` methods.
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match: File "...", line N, in func_name
        match = re.match(r'^File "(.+)", line (\d+), in (.+)$', line)
        if match:
            filename = match.group(1)
            lineno = int(match.group(2))
            function_name = match.group(3)
            # Next line (if present) is the source code
            code_context = ""
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith("File "):
                code_context = lines[i + 1].strip()
                i += 1
            if function_name == _FORWARD_FUNCTION_NAME:
                frames.append(
                    _SourceFrame(
                        filename=filename,
                        lineno=lineno,
                        function_name=function_name,
                        code_context=code_context,
                    )
                )
        i += 1
    return tuple(frames)


def _get_or_create_child(
    parent: _ModuleSummary, module_name: str, module_type: str
) -> _ModuleSummary:
    """Get an existing child module or create a new one."""
    if module_name not in parent.child_modules:
        parent.child_modules[module_name] = _ModuleSummary(
            module_name=module_name,
            module_type=module_type,
            child_modules={},
            ops=[],
            input_ops=[],
            output_ops=[],
        )
    return parent.child_modules[module_name]


def _populate_boundary_ops(module: _ModuleSummary, get_attr_names: set[str]) -> None:
    """Recursively populate ``input_ops`` and ``output_ops`` for a module tree."""
    for child in module.child_modules.values():
        _populate_boundary_ops(child, get_attr_names)

    all_subtree_ops = module.all_ops()
    subtree_op_names = {op.op_name for op in all_subtree_ops}
    ignore_names = subtree_op_names | get_attr_names

    module.input_ops = [
        op for op in all_subtree_ops if any(inp.op_name not in ignore_names for inp in op.inputs)
    ]
    module.output_ops = [
        op
        for op in all_subtree_ops
        if not op.outputs or any(out.op_name not in subtree_op_names for out in op.outputs)
    ]


def parse_ops_in_graph(model: torch.fx.GraphModule) -> _ModelOpSummary:
    """Discover all operations in a graph exported model.

    Args:
        model: An exported ``torch.fx.GraphModule`` (from ``torch.export``).

    Returns:
        A :class:`ModelSummary` with operations nested in a
        :class:`ModuleInfo` tree mirroring the ``nn.Module`` hierarchy.
    """
    # Phase 1: Walk graph and create stub OpInfo (empty inputs/outputs) for every op node.
    ops_by_name: dict[str, _OpInfo] = {}
    get_attr_names: set[str] = set()
    node_op_list: list[tuple[torch.fx.Node, _OpInfo]] = []

    for node in model.graph.nodes:
        if node.op == "get_attr":
            get_attr_names.add(node.name)

        op_type = _get_node_type(node, warn_on_failure=False)
        module_stack = _extract_module_stack(node)
        source_frames = _parse_stack_trace(node.meta.get("stack_trace"))

        op_info = _OpInfo(
            op_name=node.name,
            op_type=op_type,
            module_stack=module_stack,
            source_frames=source_frames,
            inputs=(),
            outputs=(),
        )
        ops_by_name[node.name] = op_info
        node_op_list.append((node, op_info))

    # Phase 2: Fill in inputs/outputs and build the module tree.
    root = _ModuleSummary(
        module_name="",
        module_type="",
        child_modules={},
        ops=[],
        input_ops=[],
        output_ops=[],
    )

    for node, op_info in node_op_list:
        inputs = tuple(
            ops_by_name[inp.name] for inp in node.all_input_nodes if inp.name in ops_by_name
        )
        outputs = tuple(ops_by_name[user.name] for user in node.users if user.name in ops_by_name)
        op_info.inputs = inputs
        op_info.outputs = outputs

        # Walk down the module stack, placing the op in the deepest module.
        if op_info.module_stack:
            current = root
            for ctx in op_info.module_stack:
                if ctx.module_name == "":
                    # Root module — update type info
                    if not root.module_type:
                        root.module_type = ctx.module_type
                    continue
                current = _get_or_create_child(current, ctx.module_name, ctx.module_type)

            current.ops.append(op_info)

    _populate_boundary_ops(root, get_attr_names)
    return _ModelOpSummary(model=root, mode=ExecutionMode.GRAPH)


def _filter_module_tree(module: _ModuleSummary, keep_names: set[str]) -> _ModuleSummary:
    """Recursively filter a ``ModuleInfo`` tree, keeping only matching ops."""
    filtered_children = {
        fqn: _filter_module_tree(child, keep_names) for fqn, child in module.child_modules.items()
    }
    filtered_ops = [op for op in module.ops if op.op_name in keep_names]

    return _ModuleSummary(
        module_name=module.module_name,
        module_type=module.module_type,
        child_modules=filtered_children,
        ops=filtered_ops,
        input_ops=module.input_ops,
        output_ops=module.output_ops,
    )


def filter_by_compressor(
    summary: _ModelOpSummary,
    compressor: type[_BaseModelCompressor] | None,
    gm: torch.fx.GraphModule,
) -> _ModelOpSummary:
    """Filter a summary to ops supported by the given compressor.

    Uses FX graph pattern matching to determine which ops the compressor
    can target.

    Args:
        summary: The full (unfiltered) op summary.
        compressor: A compressor class. When ``None``, returns unchanged.
        gm: The exported graph module, used for pattern matching.

    Returns:
        A filtered :class:`ModelSummary`.
    """
    if compressor is None:
        return summary

    compressible_names = compressor.get_compressible_op_names(gm, ExecutionMode.GRAPH)

    filtered_root = _filter_module_tree(summary.model, compressible_names)
    return _ModelOpSummary(model=filtered_root, mode=summary.mode)
