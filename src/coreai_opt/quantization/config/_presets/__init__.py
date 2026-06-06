# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Quantization preset package."""

from coreai_opt.quantization.config._presets.module_quantizer_config import (
    _ModuleQuantizerConfigPresets,
)
from coreai_opt.quantization.config._presets.quantizer_config import (
    _QuantizerConfigPresets,
)

__all__ = [
    "_ModuleQuantizerConfigPresets",
    "_QuantizerConfigPresets",
]
