# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Prepare a pruned model for export."""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from coreai_opt._utils.import_utils import lazy_import_coreai_torch
from coreai_opt._utils.metadata_utils import CompressionType, MILCompressionMetadata

from .spec.prune import PruneImplBase


def _process_weight_sparsification(model: nn.Module) -> None:
    """Process pruning parameterization to replace with corresponding sparse parameterization.

    Process pruning parameterization to replace with corresponding sparse module
    parameterization for Core AI export. This would in turn use the
    coreai::sparse_to_dense custom op during export.

    Args:
        model (nn.Module): The pruned model with ``PruneImplBase`` parametrizations.

    """

    def _import_coreai_torch_modules() -> tuple:
        from coreai_torch._compression.custom_layers import SparseModule  # noqa: PLC0415
        from coreai_torch._compression.utils import wrap_for_parametrization  # noqa: PLC0415

        return SparseModule, wrap_for_parametrization

    SparseModule, wrap_for_parametrization = lazy_import_coreai_torch(_import_coreai_torch_modules)

    SparseParametrization = wrap_for_parametrization(SparseModule)
    prune_mod_to_sparse_mod: dict[int, nn.Module] = {}

    for name, module in model.named_modules():
        if not isinstance(module, PruneImplBase):
            continue

        param_name = name.rsplit(".", 1)[0] + ".original"
        dense_weight = model.get_parameter(param_name).detach()

        mask = module.mask.to(torch.bool)
        nonzero_data = dense_weight[mask].flatten()

        sparse_mod = SparseParametrization(nonzero_data, mask)
        prune_mod_to_sparse_mod[id(module)] = sparse_mod

    for module in model.modules():
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in module.parametrizations.items():
            for idx, p in enumerate(parametrizations):
                if isinstance(p, PruneImplBase) and id(p) in prune_mod_to_sparse_mod:
                    module.parametrizations[param_name][idx] = prune_mod_to_sparse_mod[id(p)]


def prepare_for_mlir_export(model: nn.Module) -> nn.Module:
    """Prepare a pruned model for Core AI export.

    Removes pruning parametrizations and replaces them with sparse module
    parametrizations needed for Core AI export.

    Args:
        model (nn.Module): The pruned model to be prepared for export.

    Returns:
        nn.Module: The model modified in-place for Core AI export.
    """
    _process_weight_sparsification(model)
    return model


def prepare_for_mil_export(model: nn.Module) -> nn.Module:
    """
    Prepare a pruned model for CoreML export by removing any pruning parameterizations
    and attaching the necessary metadata needed to identify sparsified parameters.


    Args:
        model (nn.Module): The pruned model to be prepared for export.

    Returns:
        nn.Module: The model modified in-place for CoreML export.
    """
    for module in model.modules():
        if not P.is_parametrized(module):
            continue
        for param_name in list(module.parametrizations):
            has_pruning = any(
                isinstance(p, PruneImplBase) for p in module.parametrizations[param_name]
            )
            if has_pruning:
                P.remove_parametrizations(module, param_name, leave_parametrized=True)
                metadata = MILCompressionMetadata(
                    param_name=param_name,
                    compression_type=CompressionType.PRUNING,
                )
                metadata.register(module)

    MILCompressionMetadata.register_version(model)
    return model
