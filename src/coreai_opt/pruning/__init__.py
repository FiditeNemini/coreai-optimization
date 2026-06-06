# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning infrastructure for coreai_opt."""

from coreai_opt.pruning.config import (
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
)
from coreai_opt.pruning.magnitude_pruner import MagnitudePruner
from coreai_opt.pruning.spec import PruningSpec

__all__ = [
    "MagnitudePruner",
    "MagnitudePrunerConfig",
    "ModuleMagnitudePrunerConfig",
    "PruningSpec",
]
