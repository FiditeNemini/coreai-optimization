# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Preset namespace for ``KMeansPalettizerConfig``.

Accessed as ``KMeansPalettizerConfig.presets.<name>()``. Each preset returns
a fully-configured ``KMeansPalettizerConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coreai_opt.palettization.config.palettization_config import (
    ModuleKMeansPalettizerConfig,
    OpKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)

if TYPE_CHECKING:
    from coreai_opt.palettization.config.palettization_config import (
        KMeansPalettizerConfig,
    )


class _KMeansPalettizerConfigPresets:
    """Namespace exposing preset constructors for ``KMeansPalettizerConfig``.

    This class is project-internal — users access an instance through
    ``KMeansPalettizerConfig.presets``.
    """

    def __init__(self, owner_cls: type[KMeansPalettizerConfig]) -> None:
        self._owner_cls = owner_cls

    def w4(self, *, axis: int = 0, group_size: int = 16) -> KMeansPalettizerConfig:
        """4-bit palettization, per-grouped-channel, group_size defaults to 16.

        Args:
            axis (int): Channel axis to group along. Defaults to 0.
            group_size (int): Number of channels per palette group.

        Returns:
            KMeansPalettizerConfig: 4-bit palettization configuration.

        """
        weight_spec = PalettizationSpec(
            n_bits=4,
            granularity=PerGroupedChannelGranularity(axis=axis, group_size=group_size),
        )
        op_state_spec = {k: weight_spec for k in OpKMeansPalettizerConfig.get_default_state_spec()}
        global_config = ModuleKMeansPalettizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=op_state_spec,
        )
        return self._owner_cls(global_config=global_config)

    def w6(self, *, axis: int = 0, group_size: int = 16) -> KMeansPalettizerConfig:
        """6-bit palettization, per-grouped-channel, group_size defaults to 16.

        Args:
            axis (int): Channel axis to group along. Defaults to 0.
            group_size (int): Number of channels per palette group.

        Returns:
            KMeansPalettizerConfig: 6-bit palettization configuration.

        """
        weight_spec = PalettizationSpec(
            n_bits=6,
            granularity=PerGroupedChannelGranularity(axis=axis, group_size=group_size),
        )
        op_state_spec = {k: weight_spec for k in OpKMeansPalettizerConfig.get_default_state_spec()}
        global_config = ModuleKMeansPalettizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=op_state_spec,
        )
        return self._owner_cls(global_config=global_config)

    def w8(self) -> KMeansPalettizerConfig:
        """8-bit palettization, per-tensor.

        Returns:
            KMeansPalettizerConfig: 8-bit palettization configuration.

        """
        weight_spec = PalettizationSpec(
            n_bits=8,
            granularity=PerTensorGranularity(),
        )
        op_state_spec = {k: weight_spec for k in OpKMeansPalettizerConfig.get_default_state_spec()}
        global_config = ModuleKMeansPalettizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=op_state_spec,
        )
        return self._owner_cls(global_config=global_config)
