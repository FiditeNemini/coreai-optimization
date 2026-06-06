# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Palettization specs, granularity classes, and factory functions."""

from .granularity import (
    PalettizationGranularity,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)
from .spec import PalettizationSpec, default_weight_palettization_spec

__all__ = [
    "PalettizationGranularity",
    "PalettizationSpec",
    "PerGroupedChannelGranularity",
    "PerTensorGranularity",
    "default_weight_palettization_spec",
]
