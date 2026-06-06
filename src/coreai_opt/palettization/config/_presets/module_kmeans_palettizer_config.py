# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Preset namespace for ``ModuleKMeansPalettizerConfig``.

Accessed as ``ModuleKMeansPalettizerConfig.presets.<name>()``. Each preset
returns a ``ModuleKMeansPalettizerConfig`` ready to pass to
``set_module_type`` or ``set_module_name``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coreai_opt.palettization.config.palettization_config import (
    OpKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)

if TYPE_CHECKING:
    from coreai_opt.palettization.config.palettization_config import (
        ModuleKMeansPalettizerConfig,
    )


class _ModuleKMeansPalettizerConfigPresets:
    """Namespace exposing preset constructors for ``ModuleKMeansPalettizerConfig``.

    Module-level presets return a ``ModuleKMeansPalettizerConfig`` suitable for
    passing directly to ``set_module_type`` or ``set_module_name``.

    This class is project-internal — users access an instance through
    ``ModuleKMeansPalettizerConfig.presets``.
    """

    def __init__(self, owner_cls: type[ModuleKMeansPalettizerConfig]) -> None:
        self._owner_cls = owner_cls

    def w4(
        self,
        *,
        axis: int = 0,
        group_size: int = 16,
    ) -> ModuleKMeansPalettizerConfig:
        """4-bit palettization, per-grouped-channel, group_size defaults to 16.

        Args:
            axis (int): Channel axis to group along. Defaults to 0.
            group_size (int): Number of channels per palette group.

        Returns:
            ModuleKMeansPalettizerConfig: 4-bit palettization module configuration.

        """
        weight_spec = PalettizationSpec(
            n_bits=4,
            granularity=PerGroupedChannelGranularity(axis=axis, group_size=group_size),
        )
        op_state_spec = {k: weight_spec for k in OpKMeansPalettizerConfig.get_default_state_spec()}
        return self._owner_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=op_state_spec,
        )

    def w6(
        self,
        *,
        axis: int = 0,
        group_size: int = 16,
    ) -> ModuleKMeansPalettizerConfig:
        """6-bit palettization, per-grouped-channel, group_size defaults to 16.

        Args:
            axis (int): Channel axis to group along. Defaults to 0.
            group_size (int): Number of channels per palette group.

        Returns:
            ModuleKMeansPalettizerConfig: 6-bit palettization module configuration.

        """
        weight_spec = PalettizationSpec(
            n_bits=6,
            granularity=PerGroupedChannelGranularity(axis=axis, group_size=group_size),
        )
        op_state_spec = {k: weight_spec for k in OpKMeansPalettizerConfig.get_default_state_spec()}
        return self._owner_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=op_state_spec,
        )

    def w8(self) -> ModuleKMeansPalettizerConfig:
        """8-bit palettization, per-tensor.

        Returns:
            ModuleKMeansPalettizerConfig: 8-bit palettization module configuration.

        """
        weight_spec = PalettizationSpec(
            n_bits=8,
            granularity=PerTensorGranularity(),
        )
        op_state_spec = {k: weight_spec for k in OpKMeansPalettizerConfig.get_default_state_spec()}
        return self._owner_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=op_state_spec,
        )
