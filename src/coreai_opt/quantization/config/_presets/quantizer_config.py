# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Preset namespace for ``QuantizerConfig``.

Accessed as ``QuantizerConfig.presets.<name>()``. Each preset returns a
fully-configured ``QuantizerConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from coreai_opt.quantization.config.quantization_config import (
    ExecutionMode,
    ModuleQuantizerConfig,
)
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    QuantizationScheme,
    QuantizationSpec,
)

if TYPE_CHECKING:
    from coreai_opt.quantization.config.quantization_config import QuantizerConfig


class _QuantizerConfigPresets:
    """Namespace exposing preset constructors for ``QuantizerConfig``.

    This class is project-internal — users access an instance through
    ``QuantizerConfig.presets``.
    """

    def __init__(self, owner_cls: type[QuantizerConfig]) -> None:
        self._owner_cls = owner_cls

    def w8(
        self,
        *,
        axis: int | None = None,
        execution_mode: ExecutionMode = ExecutionMode.GRAPH,
    ) -> QuantizerConfig:
        """int8 weight-only quantization, per-channel symmetric.

        Args:
            axis (int | None): Channel axis for per-channel quantization.
                When ``None`` (default), the axis is auto-resolved based on the module type
                during quantization.
            execution_mode (ExecutionMode): Quantization execution mode.
                Defaults to ``ExecutionMode.GRAPH``.

        Returns:
            QuantizerConfig: int8 weight-only configuration.

        """
        weight_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerChannelGranularity(axis=axis),
        )
        global_config = ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec={"weight": weight_spec},
        )
        return self._owner_cls(
            global_config=global_config,
            execution_mode=execution_mode,
        )

    def w4(
        self,
        *,
        axis: int | None = None,
        execution_mode: ExecutionMode = ExecutionMode.GRAPH,
    ) -> QuantizerConfig:
        """int4 weight-only quantization, per-channel symmetric.

        Args:
            axis (int | None): Channel axis for per-channel quantization.
                When ``None`` (default), the axis is auto-resolved based on the module type
                during quantization.
            execution_mode (ExecutionMode): Quantization execution mode.
                Defaults to ``ExecutionMode.GRAPH``.

        Returns:
            QuantizerConfig: int4 weight-only configuration.

        """
        weight_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerChannelGranularity(axis=axis),
        )
        global_config = ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec={"weight": weight_spec},
        )
        return self._owner_cls(
            global_config=global_config,
            execution_mode=execution_mode,
        )

    def w4_per_block(
        self,
        *,
        block_size: int = 32,
        axis: int | None = None,
        execution_mode: ExecutionMode = ExecutionMode.GRAPH,
    ) -> QuantizerConfig:
        """int4 weight-only quantization, per-block symmetric, block_size defaults to 32.

        Args:
            block_size (int): Block size along the input channel dimension (default 32).
            axis (int | None): Axis to apply blocks along.
                When ``None`` (default), the axis is auto-resolved based on the module type
                during quantization.
            execution_mode (ExecutionMode): Quantization execution mode.
                Defaults to ``ExecutionMode.GRAPH``.

        Returns:
            QuantizerConfig: int4 per-block weight-only configuration.

        """
        weight_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerBlockGranularity(axis=axis, block_size=block_size),
        )
        global_config = ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec={"weight": weight_spec},
        )
        return self._owner_cls(
            global_config=global_config,
            execution_mode=execution_mode,
        )
