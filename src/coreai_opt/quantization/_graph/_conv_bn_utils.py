# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Conv+BN pattern utilities for PT2E quantization.

This module provides utilities for working with Conv+BatchNorm patterns in PT2E
quantization, including pattern matching, batch norm folding, and dtype correction
for non-float32 models.
"""

from __future__ import annotations

import logging
import operator
from typing import TypedDict

import torch
from torch.fx import GraphModule, Node

from ._utils import assign_attr, resolve_attr

logger = logging.getLogger(__name__)


# Constants for batch_norm argument indices
_BN_ARG_WEIGHT = 1
_BN_ARG_BIAS = 2
_BN_ARG_RUNNING_MEAN = 3
_BN_ARG_RUNNING_VAR = 4
_BN_ARG_MIN_COUNT = 5

# Constants for batch_norm epsilon extraction
_BN_DEFAULT_EPS_ARG_IDX = 7
_BN_NATIVE_EPS_ARG_IDX = 6

# Constants for conv argument indices
# All conv ops have the following signature:
# input, weight, bias, stride, padding, dilation, groups
# Starting from `bias`, all fields are optional
_CONV_ARG_WEIGHT = 1
_CONV_ARG_BIAS = 2
_CONV_ARG_MIN_WITH_WEIGHT = 2
_CONV_ARG_MIN_WITH_BIAS = 3


class _BatchNormParameters(TypedDict):
    """Batch normalization parameters extracted from a batch_norm node."""

    weight: torch.Tensor
    bias: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor
    eps: float


def _is_batch_norm_node(node: Node) -> bool:
    """Check if node is a batch normalization operation.

    Args:
        node: The FX graph node to check

    Returns:
        True if node is a batch norm operation

    """
    return node.op == "call_function" and node.target in (
        # Standard batch normalization operation
        torch.ops.aten.batch_norm.default,
        # Optimized batch norm for inference mode, when it knows training=False
        torch.ops.aten._native_batch_norm_legit_no_training.default,  # noqa: SLF001
    )


def _is_conv_node(node: Node) -> bool:
    """Check if node is a convolution operation.

    Args:
        node: The FX graph node to check

    Returns:
        True if node is a conv operation

    """
    return node.op == "call_function" and node.target in (
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv3d.default,
    )


def _is_conv_transpose_node(node: Node) -> bool:
    """Check if node is a transpose convolution operation.

    Args:
        node: The FX graph node to check

    Returns:
        True if node is a conv transpose operation

    """
    return node.op == "call_function" and node.target in (
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d.input,
        torch.ops.aten.conv_transpose3d.input,
    )


def _is_conv_or_conv_transpose_node(node: Node) -> bool:
    """Check if node is a conv or conv_transpose operation.

    Args:
        node: The FX graph node to check

    Returns:
        True if node is a conv or conv_transpose operation

    """
    return _is_conv_node(node) or _is_conv_transpose_node(node)


def _get_tensor_from_node(
    node: Node | None,
    model: GraphModule,
) -> torch.Tensor | None:
    """Extract tensor from a get_attr node.

    Args:
        node: The node to extract tensor from (should be get_attr)
        model: The graph module containing the tensor

    Returns:
        The tensor if node is get_attr, None if node is None

    Raises:
        TypeError: If node is not a Node type (when not None)
        ValueError: If node is not a get_attr node

    """
    if node is None:
        return None

    # Runtime type check for arguments that come from FX node.args (typed as Any)
    if not isinstance(node, Node):
        msg = f"Expected Node or None, got {type(node).__name__}"
        raise TypeError(msg)

    if node.op != "get_attr":
        msg = f"Expected get_attr node, got {node.op} for node {node.name}"
        raise ValueError(msg)

    return resolve_attr(model, str(node.target))


def _get_first_input_node_if_matches(node: Node, target: object) -> Node | None:
    """Get first input node if node is a call_function matching the target.

    Args:
        node: The node to check
        target: The expected function target (e.g., torch.ops.aten.add.Tensor)

    Returns:
        The input node (first argument) if conditions match, None otherwise

    """
    if node.op != "call_function" or node.target != target:
        return None
    if not node.args or not isinstance(node.args[0], Node):
        return None
    return node.args[0]


def _get_attr_node_if_matches(node: Node, attr_name: str) -> Node | None:
    """Get node if it's a get_attr operation for a specific attribute.

    Args:
        node: The FX graph node to check
        attr_name: Expected attribute name substring (e.g., "weight", "bias")

    Returns:
        The node if it's get_attr and target contains attr_name, None otherwise

    """
    if node.op == "get_attr" and attr_name in str(node.target):
        return node
    return None


def _find_conv_for_bn(bn_node: Node) -> Node | None:
    """Trace backwards from batch_norm to find the conv operation.

    Standard batch norm after conv computes:
        output = gamma * (conv(x, W) - mean) / sqrt(var + eps) + beta

    In the fused pattern created by prepare_qat_pt2e, BN is decomposed into TWO paths
    that together implement the complete BN operation (gamma applied exactly ONCE):

    PATH 1 - Weight pre-scaling (found by _find_conv_weight_node):
        conv_weight → reshape → mul → activation_post_process → conv
                                ↑
                        W' = W * gamma / sqrt(var + eps)
                        (folding gamma into weights)

    PATH 2 - Activation normalization (this function traces backwards through):
        Two variants exist depending on whether conv has bias:

        VARIANT A - Conv with bias (e.g., default nn.Conv2d):
            conv → div → add → batch_norm
                   ↑     ↑     ↑
            div: normalize by sqrt(var + eps)
            add: center by mean and add beta (incorporates conv bias)
            batch_norm: track running statistics (gamma already applied in weights)

        VARIANT B - Conv without bias (e.g., ResNet, where Conv2d(bias=False)):
            conv → div → batch_norm
                   ↑     ↑
            div: normalize by sqrt(var + eps)
            batch_norm: center by mean and add beta (no conv bias to incorporate)

        Why the difference: When conv has no bias, the add node is unnecessary.
        ResNet architectures use bias=False in conv layers followed by BN because
        the BN bias term makes the conv bias redundant.

    COMBINED RESULT (PATH 1 + PATH 2):
        output = batch_norm([add](div(conv(x, W * gamma / sqrt(var + eps)))))
               = gamma * (conv(x, W) - mean) / sqrt(var + eps) + beta

        This is mathematically equivalent to standard batch norm after conv.

    IMPORTANT: When mul applies gamma to weights, batch_norm.weight is neutral
    (ones/None).
    The gamma scaling happens exactly once, either in the weight path OR in batch_norm,
    never both non-trivially.

    Folding eliminates BOTH paths by baking the complete BN transformation into conv
    weights and bias.

    We need to traverse backwards through [add] and div to find conv, where add is
    optional depending on whether conv has bias.

    Args:
        bn_node: The batch_norm node

    Returns:
        The conv node if found, None otherwise

    """
    # Traverse: batch_norm <== [add <==] div <== conv
    # The add node is optional - present when conv has bias, absent when conv.bias=False

    # batch_norm always has args: (input, weight, bias, running_mean, running_var, ...)
    # bn_node.args[0] is the input to batch_norm
    if not bn_node.args or not isinstance(bn_node.args[0], Node):
        return None

    input_node = bn_node.args[0]

    # Try pattern 1: batch_norm <== add <== div <== conv (conv has bias)
    div_node = _get_first_input_node_if_matches(input_node, torch.ops.aten.add.Tensor)
    if div_node is not None:
        # Found add node, now look for div
        conv_node = _get_first_input_node_if_matches(
            div_node,
            torch.ops.aten.div.Tensor,
        )
        if conv_node is not None and _is_conv_or_conv_transpose_node(conv_node):
            return conv_node

    # Try pattern 2: batch_norm <== div <== conv (conv.bias=False, e.g., ResNet)
    conv_node = _get_first_input_node_if_matches(input_node, torch.ops.aten.div.Tensor)
    if conv_node is not None and _is_conv_or_conv_transpose_node(conv_node):
        return conv_node

    return None


def _skip_node_if_matches(
    current: Node,
    op: str,
    target: object | None = None,
    name_contains: str | None = None,
) -> Node:
    """Skip current node if it matches criteria, returning its first argument.

    Args:
        current: The node to potentially skip
        op: Required op type (e.g., "call_function", "call_module")
        target: Optional target to match (for call_function nodes)
        name_contains: Optional substring that must be in node name

    Returns:
        The first argument if node matches and has args, otherwise current node

    Raises:
        ValueError: If node matches but has no input arguments

    """
    if current.op != op:
        return current
    if target is not None and current.target != target:
        return current
    if name_contains is not None and name_contains not in current.name:
        return current

    # Node matches - get its input
    if not current.args:
        msg = f"Node {current.name} matches but has no input"
        raise ValueError(msg)
    return current.args[0]


def _find_conv_weight_node(conv_node: Node) -> Node:
    """Find the weight get_attr node for a conv operation.

    Standard batch norm after conv computes:
        output = gamma * (conv(x, W) - mean) / sqrt(var + eps) + beta

    In the fused pattern created by prepare_qat_pt2e, BN is decomposed into TWO paths
    that together implement the complete BN operation (gamma applied exactly ONCE):

    PATH 1 - Weight pre-scaling (this function traces backwards through):
        conv_weight → reshape → mul → activation_post_process → conv
                      ↑         ↑     ↑
        reshape: make gamma broadcastable ([C_out] → [C_out,1,1,1])
        mul: apply BN scale to weights (W' = W * gamma / sqrt(var + eps))
        activation_post_process: fake quantization for weight statistics
            (not used by coreai-opt)

    PATH 2 - Activation normalization (found by _find_conv_for_bn):
        conv → div → add → batch_norm
               ↑     ↑     ↑
        div: normalize by sqrt(var + eps)
        add: center by mean and add beta
        batch_norm: track running statistics (gamma already applied in weights)

    COMBINED RESULT (PATH 1 + PATH 2):
        output = batch_norm(add(div(conv(x, W * gamma / sqrt(var + eps)))))
               = gamma * (conv(x, W) - mean) / sqrt(var + eps) + beta

        This is mathematically equivalent to standard batch norm after conv.

    IMPORTANT: The mul operation applies gamma/sqrt(var+eps) to WEIGHTS (not
    activations). So this folding is not what we want.
    This is weight folding - baking BN scale into conv weights before the convolution
    runs. When this happens, batch_norm.weight is neutral (ones/None) since gamma was
    already applied in the weight path.

    We need to trace back from conv.args[1] through this chain to find the
    original weight parameter, which is where we want to bake batch norm.

    Args:
        conv_node: The conv operation node

    Returns:
        The get_attr node for the conv weight

    Raises:
        ValueError: If weight node cannot be found

    """
    # All conv ops have the following signature:
    # input, weight, bias, stride, padding, dilation, groups
    # Starting from `bias`, all fields are optional
    if len(conv_node.args) < _CONV_ARG_MIN_WITH_WEIGHT:
        msg = f"Conv node {conv_node.name} has insufficient arguments"
        raise ValueError(msg)

    weight_input = conv_node.args[_CONV_ARG_WEIGHT]
    if not isinstance(weight_input, Node):
        msg = f"Conv weight input is not a Node: {weight_input}"
        raise TypeError(msg)

    # Trace back: activation_post_process → mul → reshape → conv_weight
    current = weight_input
    current = _skip_node_if_matches(
        current,
        "call_module",
        name_contains="activation_post_process",
    )
    current = _skip_node_if_matches(
        current,
        "call_function",
        target=torch.ops.aten.mul.Tensor,
    )
    current = _skip_node_if_matches(
        current,
        "call_function",
        target=torch.ops.aten.reshape.default,
    )

    weight_node = _get_attr_node_if_matches(current, "weight")
    if weight_node:
        return weight_node

    msg = (
        f"Could not find weight get_attr node for conv {conv_node.name}, "
        f"ended at {current.name} (op: {current.op}, target: {current.target})"
    )
    raise ValueError(msg)


def _find_conv_bias_node(conv_node: Node) -> Node | None:
    """Find the bias get_attr node for a conv operation.

    In the fused conv+bn pattern, bias can appear in two forms:

    CASE 1 - Bias wrapped in zeros_like (common in fused pattern):
        conv.bias (get_attr) → zeros_like → conv
             ↑                      ↑
        original parameter    zeros tensor passed to conv

        Why: During prepare_qat_pt2e fusion, BN's add operation handles bias
        application, so conv needs to run with bias=0 to avoid double-applying.
        zeros_like creates a zero tensor while preserving the original parameter
        for gradient updates during training.

    CASE 2 - Bias used directly (unfused or post-conversion):
        conv.bias (get_attr) → conv
             ↑
        bias applied directly

    This function finds the actual bias parameter (get_attr node) regardless of
    whether it's wrapped. We need the parameter node to update it with the folded
    BN transformation.

    Args:
        conv_node: The conv operation node

    Returns:
        The get_attr node for the conv bias, or None if no bias exists

    """
    # input, weight, bias, stride, padding, dilation, groups
    # Check if conv has a bias argument (args[2])
    if len(conv_node.args) < _CONV_ARG_MIN_WITH_BIAS:
        return None

    bias_input = conv_node.args[_CONV_ARG_BIAS]
    if not isinstance(bias_input, Node):
        return None

    # Skip zeros_like wrapper if present
    # CASE 1: get_attr(conv.bias) → zeros_like → conv (returns get_attr node)
    # CASE 2: get_attr(conv.bias) → conv (returns get_attr node unchanged)
    bias_node = _skip_node_if_matches(
        bias_input,
        "call_function",
        target=torch.ops.aten.zeros_like.default,
    )

    # Verify we have a valid bias parameter and return it (or None)
    return _get_attr_node_if_matches(bias_node, "bias")


def _extract_bn_parameters(
    bn_node: Node,
    model: GraphModule,
) -> _BatchNormParameters:
    """Extract batch norm parameters from batch_norm node.

    BatchNorm args structure:
        batch_norm(input, weight, bias, running_mean, running_var,
                   training, momentum, eps, ...)

    Args:
        bn_node: The batch_norm node
        model: The graph module

    Returns:
        Dictionary with keys: weight, bias, running_mean, running_var, eps

    Raises:
        ValueError: If BN parameters cannot be extracted

    """
    # batch_norm signature:
    # input, weight, bias, running_mean, running_var, training, momentum, eps, ...
    if len(bn_node.args) < _BN_ARG_MIN_COUNT:
        msg = f"Batch norm node {bn_node.name} has insufficient arguments: {len(bn_node.args)}"
        raise ValueError(msg)

    # Extract tensors from get_attr nodes (args 1-4 are the BN parameters)
    bn_weight = _get_tensor_from_node(bn_node.args[_BN_ARG_WEIGHT], model)
    bn_bias = _get_tensor_from_node(bn_node.args[_BN_ARG_BIAS], model)
    bn_running_mean = _get_tensor_from_node(bn_node.args[_BN_ARG_RUNNING_MEAN], model)
    bn_running_var = _get_tensor_from_node(bn_node.args[_BN_ARG_RUNNING_VAR], model)

    # Validate that all required tensors were extracted
    if bn_weight is None or bn_bias is None or bn_running_mean is None or bn_running_var is None:
        msg = f"Missing required BN parameters for node {bn_node.name}"
        raise ValueError(msg)

    # Extract epsilon - can be in three places depending on batch_norm variant
    # Priority 1: kwargs (most explicit, used when eps is passed as keyword argument)
    # Priority 2: positional args (position depends on operation type)
    #   - batch_norm.default: eps at args[7]
    #   - _native_batch_norm_legit_no_training.default: eps at args[6]
    bn_eps = 1e-5  # default value if not found

    if "eps" in bn_node.kwargs:
        # Priority 1: Found in kwargs
        bn_eps = bn_node.kwargs["eps"]

    elif bn_node.target == torch.ops.aten._native_batch_norm_legit_no_training.default:  # noqa: SLF001
        # Priority 2: Native variant has eps at index 6
        if len(bn_node.args) > _BN_NATIVE_EPS_ARG_IDX:
            bn_eps = bn_node.args[_BN_NATIVE_EPS_ARG_IDX]
    elif (
        bn_node.target == torch.ops.aten.batch_norm.default
        and len(bn_node.args) > _BN_DEFAULT_EPS_ARG_IDX
    ):
        # Priority 2: Default variant has eps at index 7
        bn_eps = bn_node.args[_BN_DEFAULT_EPS_ARG_IDX]

    result: _BatchNormParameters = {
        "weight": bn_weight,
        "bias": bn_bias,
        "running_mean": bn_running_mean,
        "running_var": bn_running_var,
        "eps": float(bn_eps),
    }

    return result


def _compute_fused_conv_bn_params(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    bn_eps: float,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse batch norm parameters into conv weight and bias.

    Mirrors the fp32 arithmetic order used by the conv+bn pattern that
    ``prepare_qat_pt2e`` inserts: ``scale = bn_w / sqrt(bn_var + eps)``.
    This ensures prepare and finalized models both produce bit-identical values
    feeding into weight FQ layer leading to FQ outputs being bit-identical as well.

    We avoid using ``torch.nn.utils.fusion.fuse_conv_bn_weights`` as it uses ``rsqrt``
    instead, which differs by ~1 ulp per channel from fused graph value and can get
    amplified into a full quant step when a value sits near a quant bin boundary.
    The error then accumulates across deep models causing larger SNR differences between
    prepared and finalized model.

    Args:
        conv_w (torch.Tensor): Conv weight, shape ``[C_out, C_in/groups, *K]``
            (or transposed equivalent when ``transpose=True``).
        conv_b (torch.Tensor): Conv bias, shape ``[C_out]``. Pass zeros if
            the conv has no bias.
        bn_running_mean (torch.Tensor): BN running mean, shape ``[C_out]``.
        bn_running_var (torch.Tensor): BN running variance, shape ``[C_out]``.
        bn_eps (float): BN epsilon.
        bn_weight (torch.Tensor): BN gamma, shape ``[C_out]``.
        bn_bias (torch.Tensor): BN beta, shape ``[C_out]``.
        transpose (bool): True if ``conv_w`` is a conv-transpose weight.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(fused_weight, fused_bias)`` with
        the same shapes and dtypes as ``conv_w`` and ``conv_b``. BN params are
        commonly fp32 even when conv params are fp16/bf16; fp32 BN params would
        promote the result, so the fused tensors are cast back to the conv
        dtypes before returning.
    """
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype
    shape = ([1, -1] if transpose else [-1, 1]) + [1] * (len(conv_w.shape) - 2)
    scale = bn_weight / torch.sqrt(bn_running_var + bn_eps)
    fused_weight = (conv_w * scale.reshape(shape)).to(dtype=conv_weight_dtype)
    fused_bias = ((conv_b - bn_running_mean) * scale + bn_bias).to(dtype=conv_bias_dtype)
    return fused_weight, fused_bias


def fold_conv_bn_weights(model: GraphModule) -> GraphModule:
    """Fold batch norm weights into conv/conv_transpose weights.

    Identifies conv+bn patterns produced by ``prepare_qat_pt2e`` and folds the
    batch-norm parameters into the convolution weights so the batch-norm
    operation can be removed.

    Folding applies the mathematical transformation::

        fused_weight = (bn_weight / sqrt(bn_var + eps)) * conv_weight
        fused_bias   = (bn_weight / sqrt(bn_var + eps)) * (conv_bias - bn_mean) + bn_bias

    Graph structure (prepared, post-``prepare_qat_pt2e``)
    ----------------------------------------------------

    ``prepare_qat_pt2e`` decomposes conv+bn into two cooperating paths so that
    the weight observer sees the BN-scaled weight while the network output
    still equals plain ``BN(conv(x))``. Both paths share one computed
    ``scale = bn_w / sqrt(bn_var + eps)`` node, reshaped two different ways::

        # Shared scale (single node in the graph, reshaped per consumer)
        bn_running_var ─→ add(eps) ─→ sqrt ─┐
                                            div  ── (this is `scale`)
        bn_weight ──────────────────────────┘

        # Weight path — pre-scale W by `scale`, then weight FQ
        scale ─→ reshape([-1, 1, 1, 1]) ─┐
                                            mul ─→ weight_FQ
        conv_w ────────────────────────────┘

        # Activation path — undo the weight pre-scale, then real BN
                    weight_FQ    scale ─→ reshape([1, -1, 1, 1])
                        │                    │
                        ▼                    ▼
        x ─→ act_FQ ─→ conv2d ──────────→ div_act ─→ [add(conv_b)] ─→ batch_norm ─→ act_FQ_out

      • Variant A (conv has bias): ``conv → div_act → add → batch_norm``
      • Variant B (conv.bias=False, e.g. ResNet): ``conv → div_act → batch_norm``

      Together the two paths compute ``BN(conv(x))`` exactly: the ``mul``
      bakes ``scale`` into the conv weight (so the weight FQ observes it),
      ``div_act`` undoes that scaling on the activation side, and
      ``batch_norm`` then re-applies ``bn_w``/``bn_b`` along with the mean
      correction.

    Graph structure (finalized, post-fold)
    --------------------------------------

    After folding, the BN operation, the weight-side ``mul``/``reshape``/
    ``div``-of-bn-params chain, and the activation-side ``div``/``add``/
    ``batch_norm`` chain all become dead code. Each conv+bn pair collapses
    to a single conv with pre-baked weight and bias::

        x ─→ act_FQ ─────────────────────────────────────────┐
                                                             ▼
        conv_w (now stores fused_weight) ─→ weight_FQ ──→ conv2d(bias=fused_bias) ─→ act_FQ_out

      • The weight stored at ``conv.weight`` becomes ``fused_weight``.
      • The conv now uses ``fused_bias`` directly (no ``zeros_like`` wrapper);
        if the conv was created with ``bias=False`` a bias parameter is added.
      • The ``weight_FQ`` and ``act_FQ`` modules are preserved bit-for-bit
        (same instances, same scale/zero-point) — only their inputs are
        rewired. Because ``fused_weight`` is computed with the same fp32 op
        order as the prepared graph's ``mul``, the weight FQ sees the same
        input in both graphs.

    Implementation steps
    --------------------

    1. Find ``batch_norm`` nodes in the graph.
    2. Trace backwards through the activation path to find the associated
       conv operation.
    3. Extract conv and BN parameters from the graph.
    4. Compute fused weight/bias via :func:`_compute_fused_conv_bn_params`,
       which mirrors the prepared graph's exact arithmetic (``bn_w /
       sqrt(bn_var + eps)``).
    5. Update the conv weight parameter with ``fused_weight``.
    6. Rewire the graph: bypass the weight-side ``mul`` (so the weight FQ
       reads the fused weight directly), and add/wire the ``fused_bias`` on
       the conv.
    7. Replace BN-node uses with the conv's output, dropping the
       activation-side ``div``/``add``/``batch_norm``.
    8. Run dead-code elimination and recompile.

    Args:
        model: The graph module after ``convert_pt2e()``.

    Returns:
        The graph module with batch norm folded into conv weights.

    Raises:
        ValueError: If pattern detection or parameter extraction fails.

    """
    # If no batch_norm nodes, return early
    has_bn = any(_is_batch_norm_node(n) for n in model.graph.nodes)
    if not has_bn:
        logger.info("No batch_norm nodes found, skipping conv+bn folding")
        return model

    folded_count = 0

    # Find all batch_norm nodes and attempt to fold them
    for bn_node in list(model.graph.nodes):
        if not _is_batch_norm_node(bn_node):
            continue

        # Trace back to find the conv operation
        conv_node = _find_conv_for_bn(bn_node)
        if conv_node is None:
            msg = f"Could not find conv for batch_norm node {bn_node.name}, skipping"
            logger.debug(msg)
            continue

        try:
            # Find conv weight and bias nodes
            conv_weight_node = _find_conv_weight_node(conv_node)
            conv_bias_node = _find_conv_bias_node(conv_node)

            # Extract parameters
            conv_w = _get_tensor_from_node(conv_weight_node, model)
            conv_b = _get_tensor_from_node(conv_bias_node, model)
            bn_params = _extract_bn_parameters(bn_node, model)

            # Validate conv weight exists (required)
            if conv_w is None:
                msg = f"Conv weight is None for node {conv_node.name}"
                raise ValueError(msg)

            # If conv has no bias, provide zeros (PyTorch convention)
            if conv_b is None:
                conv_b = torch.zeros(
                    bn_params["running_mean"].shape,
                    device=bn_params["running_mean"].device,
                    dtype=bn_params["running_mean"].dtype,
                )

            transpose = _is_conv_transpose_node(conv_node)
            fused_weight, fused_bias = _compute_fused_conv_bn_params(
                conv_w,
                conv_b,
                bn_params["running_mean"],
                bn_params["running_var"],
                bn_params["eps"],
                bn_params["weight"],
                bn_params["bias"],
                transpose=transpose,
            )

            # Update conv weight parameter
            weight_attr_name = str(conv_weight_node.target)
            assign_attr(model, weight_attr_name, fused_weight)

            # Rewire the graph to bypass BN scaling in the weight path
            # The fused pattern has:
            #   conv_weight → mul (BN scale) → activation_post_process → conv
            # After folding, we want:
            #   conv_weight (fused) → activation_post_process → conv
            mul_node = next(
                (
                    user
                    for user in conv_weight_node.users
                    if user.op == "call_function" and user.target == torch.ops.aten.mul.Tensor
                ),
                None,
            )
            if mul_node:
                mul_node.replace_all_uses_with(conv_weight_node)

            # Handle bias parameter
            if conv_bias_node:
                bias_attr_name = str(conv_bias_node.target)
            else:
                # Create bias if it doesn't exist
                bias_attr_name = weight_attr_name.replace("weight", "bias")
                # Add bias get_attr node to graph before conv
                with model.graph.inserting_before(conv_node):
                    conv_bias_node = model.graph.get_attr(bias_attr_name)

            # Update conv args to use bias directly (replacing zeros_like if present)
            conv_args = list(conv_node.args)
            if len(conv_args) >= _CONV_ARG_MIN_WITH_BIAS:
                conv_args[_CONV_ARG_BIAS] = conv_bias_node
            else:
                conv_args.append(conv_bias_node)
            conv_node.args = tuple(conv_args)

            # Update bias parameter with fused value
            assign_attr(model, bias_attr_name, fused_bias)

            # Replace BN node with conv's output
            # Handle different BN op types
            if bn_node.target == torch.ops.aten.batch_norm.default:
                # batch_norm.default has single output
                bn_node.replace_all_uses_with(conv_node)
            else:
                # _native_batch_norm_legit_no_training has 3 outputs (out, mean, var)
                # Replace uses of getitem[0] with conv output
                for user in list(bn_node.users):
                    if (
                        user.op == "call_function"
                        and user.target == operator.getitem
                        and user.args[1] == 0
                    ):
                        user.replace_all_uses_with(conv_node)

            folded_count += 1

        # Catch all exceptions to continue folding other patterns
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to fold batch_norm {bn_node.name} into conv {conv_node.name}: {e}"
            logger.warning(msg)
            continue

    # Clean up dead nodes (BN nodes and their parameters)
    # Dead code elimination removes:
    # 1. The mul node that applied BN scaling (replaced by conv_weight_node via
    #    rewiring)
    # 2. The batch_norm node itself (replaced by conv output)
    # 3. The div and optionally add nodes in activation path (replaced by conv output)
    # 4. Any other nodes that are no longer used after rewiring
    # Note: conv_weight_node itself is NOT removed - it stays in the graph but now
    # references the fused weights (the parameter at its target path was updated)
    model.graph.eliminate_dead_code()
    model.recompile()

    msg = f"Successfully folded {folded_count} conv+bn pattern(s)"
    logger.info(msg)

    return model


def remove_conv_bn_zeros_like_dtype(model: GraphModule) -> None:
    """Remove hardcoded float32 dtype from zeros_like nodes in Conv+BN patterns.

    When prepare_qat_pt2e decomposes Conv+BatchNorm patterns, it creates zeros_like
    nodes with hardcoded dtype=torch.float32 kwargs. This causes dtype mismatches when
    the model uses float16 or bfloat16.

    This function identifies zeros_like nodes that are part of Conv+BN patterns and
    removes the explicit dtype kwarg, allowing zeros_like to infer the correct dtype
    from the input tensor.

    Pattern matched:
        conv.bias (get_attr) → zeros_like → conv (in Conv+BN decomposition)

    Args:
        model: The graph module after prepare_qat_pt2e()

    """
    # Find all batch_norm nodes first to identify Conv+BN patterns
    modified = False
    for bn_node in model.graph.nodes:
        if not _is_batch_norm_node(bn_node):
            continue

        # Trace back to find the conv operation
        conv_node = _find_conv_for_bn(bn_node)
        if conv_node is None:
            continue

        # Check if conv has a bias argument with zeros_like wrapper
        if len(conv_node.args) < _CONV_ARG_MIN_WITH_BIAS:
            continue

        bias_input = conv_node.args[_CONV_ARG_BIAS]
        if bias_input is None:
            continue

        # Check if this is a zeros_like node with hardcoded dtype
        if (
            bias_input.op == "call_function"
            and bias_input.target == torch.ops.aten.zeros_like.default
            and "dtype" in bias_input.kwargs
        ):
            # Remove the explicit dtype kwarg to let zeros_like infer from input
            bias_input.kwargs = {k: v for k, v in bias_input.kwargs.items() if k != "dtype"}
            modified = True

    # Recompile the graph only if nodes were modified
    if modified:
        model.recompile()
