# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared utilities for eager-mode (torch_function) compressors."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from coreai_opt._utils.config_utils import ConfigLevel
from coreai_opt._utils.insertion.torch_function import (
    ModuleCompressionComponents,
    OpCompressionComponents,
)
from coreai_opt._utils.spec_utils import PartialConstructor
from coreai_opt._utils.torch_utils import NamedModule
from coreai_opt.config.compression_config import (
    CompressionConfig,
    ModuleCompressionConfig,
    OpCompressionConfig,
)
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.config.spec.base import CompressionSpec


class EagerCompressionComponentBuilderMixin:
    """Builds compression components and module priorities for eager compressors.

    Eager compressors feed a ``TorchFunctionEagerHandler`` with two structures
    derived from a ``CompressionConfig``:

    1. A mapping from each ``NamedModule`` to its ``ModuleCompressionComponents``.
    2. A priority dict that breaks ties when a tensor is shared across modules,
       respecting the ``MODULE_NAME > MODULE_TYPE > GLOBAL`` config precedence.

    The traversal and assembly logic is identical across compressors; the only
    domain-specific step is converting a single spec into a partial constructor.
    Subclasses implement that step in :meth:`_spec_to_partial`; everything else
    is shared.
    """

    @staticmethod
    def _spec_to_partial(
        spec: CompressionSpec | None,
        target: CompressionTargetTensor,
        module_config: ModuleCompressionConfig,
    ) -> PartialConstructor | None:
        """Convert a single spec to a partial constructor for the eager handler.

        Args:
            spec (CompressionSpec | None): The spec to convert. ``None`` means
                "no compression for this tensor" and should be passed through.
            target (CompressionTargetTensor): Whether the tensor is a weight,
                activation, or LUT. Compressors that only target weights may
                ignore this.
            module_config (ModuleCompressionConfig): The owning module config
                (the *parent* for op-level specs). Compressors with per-module
                settings (e.g., ``enable_fast_kmeans_mode``) read them from here.

        Returns:
            PartialConstructor | None: Partial constructor for the corresponding
                fake-compress module, or ``None`` if ``spec`` is ``None``.
        """
        raise NotImplementedError

    def _create_component_dict(
        self,
        spec_dict: Mapping[str | int, CompressionSpec | None],
        target: CompressionTargetTensor,
        module_config: ModuleCompressionConfig,
    ) -> dict[str | int, PartialConstructor | None]:
        """Convert a spec dict to a partial-constructor dict via ``_spec_to_partial``."""
        return {
            identifier: self._spec_to_partial(spec, target, module_config)
            for identifier, spec in spec_dict.items()
        }

    def _create_op_compression_components(
        self,
        op_config: OpCompressionConfig,
        module_config: ModuleCompressionConfig,
    ) -> OpCompressionComponents:
        """Convert an :class:`OpCompressionConfig` to ``OpCompressionComponents``.

        ``module_config`` is the parent module's config, which carries any
        compressor-specific settings that apply to ops within it.
        """
        return OpCompressionComponents(
            op_input_components=self._create_component_dict(
                op_config.op_input_spec, CompressionTargetTensor.ACTIVATION, module_config
            ),
            op_output_components=self._create_component_dict(
                op_config.op_output_spec, CompressionTargetTensor.ACTIVATION, module_config
            ),
            op_state_components=self._create_component_dict(
                op_config.op_state_spec, CompressionTargetTensor.WEIGHT, module_config
            ),
        )

    def _get_module_compression_components_and_priority(
        self,
        model: torch.nn.Module,
        config: CompressionConfig,
    ) -> tuple[dict[NamedModule, ModuleCompressionComponents], dict[str, int]]:
        """Build module compression components and module priorities for the model.

        Each module that has at least one configured component is included in the
        returned components dict. Priorities are assigned to every module touched
        by ``config``, with priority 0 = highest precedence. The
        ``TorchFunctionEagerHandler`` uses these to resolve which spec wins when
        a tensor is shared across modules.

        Args:
            model (torch.nn.Module): The model whose modules are inspected.
            config (CompressionConfig): The top-level compression configuration.

        Returns:
            tuple[dict[NamedModule, ModuleCompressionComponents], dict[str, int]]:
                A pair of (module -> compression components, module name -> priority).
        """
        module_config_dict = config.build_module_config_dict(model)
        module_components_dict: dict[NamedModule, ModuleCompressionComponents] = {}

        # Priority dict: each module gets a unique priority. Lower priority value
        # means higher precedence. ConfigLevel iterates highest -> lowest precedence.
        module_priority_dict: dict[str, int] = {}
        priority = 0
        for config_level in ConfigLevel.priority_order():
            for module_name in module_config_dict[config_level]:
                module_priority_dict[module_name] = priority
                priority += 1

        for name, module in model.named_modules(remove_duplicate=True):
            module_config = (
                module_config_dict[ConfigLevel.MODULE_NAME].get(name)
                or module_config_dict[ConfigLevel.MODULE_TYPE].get(name)
                or module_config_dict[ConfigLevel.GLOBAL].get(name)
            )
            assert module_config is not None, (
                f"Module name {name} not found in module_config_dict: {module_config_dict}"
            )

            compression_components = ModuleCompressionComponents(
                weight=self._create_component_dict(
                    module_config.op_state_spec, CompressionTargetTensor.WEIGHT, module_config
                ),
                input_activation=self._create_component_dict(
                    module_config.op_input_spec,
                    CompressionTargetTensor.ACTIVATION,
                    module_config,
                ),
                output_activation=self._create_component_dict(
                    module_config.op_output_spec,
                    CompressionTargetTensor.ACTIVATION,
                    module_config,
                ),
                op_type_components={
                    op_type: self._create_op_compression_components(op_type_config, module_config)
                    for op_type, op_type_config in module_config.op_type_config.items()
                },
                op_name_components={
                    op_name: self._create_op_compression_components(op_name_config, module_config)
                    for op_name, op_name_config in module_config.op_name_config.items()
                },
                module_input_components=self._create_component_dict(
                    module_config.module_input_spec,
                    CompressionTargetTensor.ACTIVATION,
                    module_config,
                ),
                module_output_components=self._create_component_dict(
                    module_config.module_output_spec,
                    CompressionTargetTensor.ACTIVATION,
                    module_config,
                ),
                module_state_components=self._create_component_dict(
                    module_config.module_state_spec,
                    CompressionTargetTensor.WEIGHT,
                    module_config,
                ),
            )

            if compression_components.has_any_component():
                module_components_dict[NamedModule(name, module)] = compression_components

        return module_components_dict, module_priority_dict
