# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MLIR type mapping and constant creation utilities for Core AI Optimization passes."""

# TODO: add test enhancements for type utils.
from __future__ import annotations

from typing import Any

from coreai_opt.coreai_utils._coreai_imports import (
    Context,
    Float4E2M1FNType,
    Float8E4M3FNType,
    Float8E5M2Type,
    Float8E8M0FNUType,
    FloatType,
    IntegerType,
    ml_dtypes,
)
from coreai_opt.coreai_utils.common import DType

_CHAR_BIT = 8

_STRING_TO_MLIR_TYPE: dict[str, IntegerType | FloatType] | None = None


def _get_string_to_mlir_type() -> dict[str, IntegerType | FloatType]:
    """Return a mapping from dtype string names to MLIR types.

    Returns:
        dict[str, IntegerType | FloatType]: Mapping of dtype strings (e.g. ``"int4"``,
            ``"fp8_e4m3fn"``) to the corresponding MLIR type objects.
    """
    global _STRING_TO_MLIR_TYPE
    if _STRING_TO_MLIR_TYPE is None:
        ctx = Context()
        _STRING_TO_MLIR_TYPE = {
            "fp4_e2m1": Float4E2M1FNType.get(context=ctx),
            "fp8_e4m3fn": Float8E4M3FNType.get(context=ctx),
            "fp8_e5m2": Float8E5M2Type.get(context=ctx),
            "fp8_e8m0fnu": Float8E8M0FNUType.get(context=ctx),
            "int2": IntegerType.get_signed(2, context=ctx),
            "int4": IntegerType.get_signed(4, context=ctx),
            "int8": IntegerType.get_signed(8, context=ctx),
            "uint2": IntegerType.get_unsigned(2, context=ctx),
            "uint4": IntegerType.get_unsigned(4, context=ctx),
            "uint8": IntegerType.get_unsigned(8, context=ctx),
        }
    return _STRING_TO_MLIR_TYPE


def _is_sub_byte_int(value_type: Any) -> bool:
    """Determine if a Type is sub-byte integer type."""
    if isinstance(value_type, IntegerType):
        if value_type.is_signless and value_type.width == 1:
            # Signless 1-bit is bool, not treated as sub-byte.
            return False
        return bool(value_type.width < _CHAR_BIT)
    return False


def _get_fp_mlir_and_ml_dtype(
    dtype: DType,
    context: Context | None = None,
) -> tuple[Float4E2M1FNType | Float8E4M3FNType | Float8E5M2Type, Any]:
    """Return the MLIR type and ml_dtypes scalar type for a float DType.

    Args:
        dtype (DType): A float dtype (``DType.FP4_E2M1FN``, ``DType.FP8_E4M3FN``,
            or ``DType.FP8_E5M2``).
        context (Context | None): The MLIR context to use. If ``None``, the
            current thread-local active context is used.

    Returns:
        tuple[Float4E2M1FNType | Float8E4M3FNType | Float8E5M2Type, Any]: A pair of
            ``(mlir_type, ml_dtype)`` corresponding to the given float dtype.

    Raises:
        ValueError: If dtype is not a supported float dtype.
    """
    if dtype == DType.FP4_E2M1FN:
        return Float4E2M1FNType.get(context=context), ml_dtypes.float4_e2m1fn
    elif dtype == DType.FP8_E4M3FN:
        return Float8E4M3FNType.get(context=context), ml_dtypes.float8_e4m3fn
    elif dtype == DType.FP8_E5M2:
        return Float8E5M2Type.get(context=context), ml_dtypes.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP dtype: {dtype}")


def _get_scale_mlir_and_np_dtype(
    scale_dtype: DType,
    context: Context | None = None,
) -> tuple[Any, Any]:
    """Return the MLIR type and numpy dtype for a scale DType.

    Args:
        scale_dtype (DType): A supported scale dtype (``DType.FP8_E8M0FNU``).
        context (Context | None): The MLIR context to use. If ``None``, the
            current thread-local active context is used.

    Returns:
        tuple[Any, Any]: A pair of ``(mlir_type, np_dtype)`` for the given scale dtype.

    Raises:
        ValueError: If scale_dtype is not a supported scale dtype.
    """
    if scale_dtype == DType.FP8_E8M0FNU:
        return Float8E8M0FNUType.get(context=context), ml_dtypes.float8_e8m0fnu
    raise ValueError(f"Unsupported scale dtype: {scale_dtype}")
