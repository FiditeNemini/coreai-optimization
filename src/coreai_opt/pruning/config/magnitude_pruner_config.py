# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning configuration classes."""

from typing import ClassVar

from coreai_opt.config import (
    CompressionConfig,
    ModuleCompressionConfig,
    OpCompressionConfig,
    WeightOnlyModuleValidationMixin,
    WeightOnlyOpValidationMixin,
)
from coreai_opt.pruning.spec import PruningSpec, default_weight_pruning_spec

from .sparsity_schedule import SparsityScheduleBase

_MAGNITUDE_PRUNING_CONFIG = "magnitude_pruning_config"
_PRUNING_SPEC = "pruning_spec"


class OpMagnitudePrunerConfig(WeightOnlyOpValidationMixin, OpCompressionConfig[PruningSpec]):
    """Operation-level pruning configuration.

    Pruning is a weight-only compression technique. Only ``op_state_spec``
    is used to configure which state tensors (e.g. weights) to prune.

    Attributes:
        op_state_spec (dict[str, PruningSpec | None]): Mapping of parameter
            names to their pruning specs. Default includes ``"weight"`` and
            ``"in_proj_weight"`` at 50 % sparsity.

    Example:
        >>> config = OpMagnitudePrunerConfig()
        >>> config = OpMagnitudePrunerConfig(
        ...     op_state_spec={"weight": PruningSpec(target_sparsity=0.75)}
        ... )
    """

    @classmethod
    def get_default_state_spec(cls) -> dict[str, PruningSpec | None]:
        """Provide default state spec for pruning."""
        spec = default_weight_pruning_spec()
        return {"weight": spec, "in_proj_weight": spec}


class ModuleMagnitudePrunerConfig(
    WeightOnlyModuleValidationMixin,
    ModuleCompressionConfig[OpMagnitudePrunerConfig, PruningSpec],
):
    """Module-level pruning configuration.

    Manages pruning settings for an entire module, following the same
    hierarchical precedence as other compression configs:

    1. ``op_name_config`` (most specific)
    2. ``op_type_config``
    3. ``op_state_spec`` (least specific)

    Attributes:
        op_state_spec (dict[str, PruningSpec | None] | None): Default pruning
            specs for state tensors in this module.
        op_type_config (dict[str, OpMagnitudePrunerConfig]): Per-op-type overrides.
        op_name_config (dict[str, OpMagnitudePrunerConfig]): Per-op-name overrides.
        module_state_spec (dict[str, PruningSpec | None] | None): Specs applied
            across all ops in the module.
        sparsity_schedule (SparsityScheduleBase | None): Optional sparsity schedule.
            When set, the ``pruner.step()`` API drives sparsity over training
            steps; when ``None`` (default), the spec's ``target_sparsity`` is
            applied immediately and statically.
    """

    sparsity_schedule: SparsityScheduleBase | None = None


class MagnitudePrunerConfig(CompressionConfig[ModuleMagnitudePrunerConfig]):
    """Top-level configuration for magnitude pruning.

    Attributes:
        global_config (ModuleMagnitudePrunerConfig | None): Default pruning
            config applied to all modules.
        module_type_configs (dict[str, ModuleMagnitudePrunerConfig | None]):
            Per-module-type overrides.
        module_name_configs (dict[str, ModuleMagnitudePrunerConfig | None]):
            Per-module-name overrides (highest priority).

    Example:
        >>> config = MagnitudePrunerConfig()  # 50 % sparsity everywhere
        >>> config = MagnitudePrunerConfig(
        ...     module_name_configs={"layer1": None}  # skip layer1
        ... )
    """

    _CONFIG_KEY: ClassVar[str] = _MAGNITUDE_PRUNING_CONFIG
    _SPEC_KEY: ClassVar[str] = _PRUNING_SPEC
