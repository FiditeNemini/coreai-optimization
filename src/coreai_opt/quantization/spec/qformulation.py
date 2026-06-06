# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Quantization formulation enum used by :class:`QuantizationSpec`."""

from enum import auto

from coreai_opt.common import _StrEnum

__all__ = ["QuantizationFormulation"]


class QuantizationFormulation(_StrEnum):
    """Formula used to map between quantized integers and dequantized values.

    Attributes:
        ZP: Standard zero-point formulation.

            - ``q  = clamp(round(x / scale) + zero_point, quant_min, quant_max)``
            - ``x' = (q - zero_point) * scale``

        MINVAL: Min-value formulation.

            - ``q  = clamp(round((x - minval) / scale) + quant_min, quant_min, quant_max)``
            - ``x' = (q - quant_min) * scale + minval``
    """

    MINVAL = auto()
    ZP = auto()
