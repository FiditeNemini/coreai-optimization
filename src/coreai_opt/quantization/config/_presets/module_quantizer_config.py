# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Preset namespace for ``ModuleQuantizerConfig``.

Accessed as ``ModuleQuantizerConfig.presets.<name>()``. Each preset returns
a ``ModuleQuantizerConfig`` ready to pass to ``set_module_type`` or
``set_module_name``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    QuantizationScheme,
    QuantizationSpec,
)

if TYPE_CHECKING:
    from coreai_opt.quantization.config.quantization_config import (
        ModuleQuantizerConfig,
    )


class _ModuleQuantizerConfigPresets:
    """Namespace exposing preset constructors for ``ModuleQuantizerConfig``.

    Module-level presets return a ``ModuleQuantizerConfig`` suitable for
    passing directly to ``set_module_type`` or ``set_module_name``.

    This class is project-internal — users access an instance through
    ``ModuleQuantizerConfig.presets``.
    """

    def __init__(self, owner_cls: type[ModuleQuantizerConfig]) -> None:
        self._owner_cls = owner_cls

    def w8(self, *, axis: int | None = None) -> ModuleQuantizerConfig:
        """int8 weight-only quantization, per-channel symmetric.

        Args:
            axis (int | None): Channel axis for per-channel quantization.
                When ``None`` (default), the axis is auto-resolved based on the module type
                during quantization.

        Returns:
            ModuleQuantizerConfig: int8 weight-only module configuration.

        """
        weight_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerChannelGranularity(axis=axis),
        )
        return self._owner_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec={"weight": weight_spec},
        )

    def w4(
        self,
        *,
        axis: int | None = None,
    ) -> ModuleQuantizerConfig:
        """int4 weight-only quantization, per-channel symmetric.

        Args:
            axis (int | None): Channel axis for per-channel quantization.
                When ``None`` (default), the axis is auto-resolved based on the module type
                during quantization.

        Returns:
            ModuleQuantizerConfig: int4 weight-only module configuration.

        """
        weight_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerChannelGranularity(axis=axis),
        )
        return self._owner_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec={"weight": weight_spec},
        )

    def w4_per_block(
        self,
        *,
        block_size: int = 32,
        axis: int | None = None,
    ) -> ModuleQuantizerConfig:
        """int4 weight-only quantization, per-block symmetric, block_size defaults to 32.

        Args:
            block_size (int): Block size along the input channel dimension (default 32).
            axis (int | None): Axis to apply blocks along.
                When ``None`` (default), the axis is auto-resolved based on the module type
                during quantization.

        Returns:
            ModuleQuantizerConfig: int4 per-block weight-only module configuration.

        """
        weight_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerBlockGranularity(axis=axis, block_size=block_size),
        )
        return self._owner_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec={"weight": weight_spec},
        )
