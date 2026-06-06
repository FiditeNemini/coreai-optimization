# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Common enums and constants for coreai_opt.coreai_utils."""

from __future__ import annotations

from enum import auto

from coreai_opt.common import _StrEnum


class DType(_StrEnum):
    """Enum representing data types for Core AI weight compression.

    Each member's string value matches the dtype string accepted by Core AI
    compression utilities (e.g. ``compression_types.string_to_builtin``).

    Attributes:
        INT2: Signed 2-bit integer.
        UINT2: Unsigned 2-bit integer.
        INT4: Signed 4-bit integer.
        UINT4: Unsigned 4-bit integer.
        INT8: Signed 8-bit integer.
        UINT8: Unsigned 8-bit integer.
        FP4_E2M1FN: 4-bit floating-point (E2M1FN).
        FP8_E4M3FN: 8-bit floating-point (E4M3FN).
        FP8_E5M2: 8-bit floating-point (E5M2).
        FP8_E8M0FNU: 8-bit floating-point with 8 exponent bits, no mantissa, no sign
            (E8M0FNU). Used as a scale dtype for FP4/FP8 quantization (MXFP format).
    """

    INT2 = auto()
    UINT2 = auto()
    INT4 = auto()
    UINT4 = auto()
    INT8 = auto()
    UINT8 = auto()
    FP4_E2M1FN = auto()
    FP8_E4M3FN = auto()
    FP8_E5M2 = auto()
    FP8_E8M0FNU = auto()

    def is_int(self) -> bool:
        """Return True if this dtype is an integer type."""
        return self in {
            DType.INT2,
            DType.INT4,
            DType.INT8,
            DType.UINT2,
            DType.UINT4,
            DType.UINT8,
        }


class QScheme(_StrEnum):
    """Enum representing the quantization scheme.

    Attributes:
        SYMMETRIC: Symmetric quantization (zero-point is fixed at zero).
        ASYMMETRIC: Asymmetric quantization (zero-point is non-zero).
    """

    SYMMETRIC = auto()
    ASYMMETRIC = auto()


class CompressionGranularity(_StrEnum):
    """Enum representing the granularity of quantization for Core AI weight compression.

    Each member's string value matches the granularity string accepted by Core AI
    compression passes.

    Attributes:
        PER_TENSOR: Single set of quantization parameters for the entire tensor.
        PER_CHANNEL: Separate quantization parameters per individual axis. The targeted axis
            is pre-defined by the type of operations.
        PER_BLOCK: Separate quantization parameters per block of axes. The targeted axes
            are pre-defined by the type of operations.
        PER_GROUPED_CHANNEL: Separate quantization parameters per group of channels.
    """

    PER_TENSOR = auto()
    PER_CHANNEL = auto()
    PER_BLOCK = auto()
    PER_GROUPED_CHANNEL = auto()


__all__ = ["CompressionGranularity", "DType", "QScheme"]
