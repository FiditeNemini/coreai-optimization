# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Eager mode compression utilities using torch function interception."""

from .base_supported_ops_registry import BaseSupportedOpsRegistry
from .handler import TorchFunctionEagerHandler
from .modes import (
    ActivationEagerOptimizationHandler,
    RegisterEagerOptimizationMode,
    ScopedEagerOptimizationModeBase,
)
from .types import (
    ActHandlerOutput,
    ModuleCompressionComponents,
    OpCompressionComponents,
)
from .utils import (
    normalize_args_kwargs,
)

__all__ = [  # noqa: RUF022
    # Ops Registry
    "BaseSupportedOpsRegistry",
    # Handler
    "TorchFunctionEagerHandler",
    # Modes
    "ScopedEagerOptimizationModeBase",
    "RegisterEagerOptimizationMode",
    "ActivationEagerOptimizationHandler",
    # Types
    "ModuleCompressionComponents",
    "OpCompressionComponents",
    "ActHandlerOutput",
    # Utils
    "normalize_args_kwargs",
]
