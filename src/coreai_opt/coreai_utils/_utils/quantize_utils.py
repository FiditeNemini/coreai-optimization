# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Quantization utilities for Core AI Optimization passes."""

from __future__ import annotations

import logging
from typing import Any

import ml_dtypes
import numpy as np

from coreai_opt.coreai_utils._coreai_imports import compression_types
from coreai_opt.coreai_utils.common import QScheme

logger = logging.getLogger(__name__)

_FLOAT_ML_DTYPES = (ml_dtypes.float4_e2m1fn, ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2)


def _is_float_ml_dtype(dtype: Any) -> bool:
    return dtype in _FLOAT_ML_DTYPES


# TODO: unify the dtypes and add proper type annotation.
def _get_quantize_range_by_dtype(
    dtype: Any,
    mode: QScheme,
) -> tuple[float, float]:
    """Return the quantization range for a given dtype and mode.

    For integer types the range is computed from the bit-width. For float ml_dtypes
    types the range is ``(-max, max)`` where ``max`` is the largest finite value of
    the dtype (always symmetric; ``mode`` is ignored for float ml_dtypes types).

    Args:
        dtype (Any): The quantization dtype. Either a coreai compression
            builtin int type or an ml_dtypes float numpy dtype.
        mode (QScheme): Quantization scheme, one of ``QScheme.SYMMETRIC`` or
            ``QScheme.ASYMMETRIC``. Ignored for float ml_dtypes types.

    Returns:
        tuple[float, float]: ``(quant_min, quant_max)`` inclusive range.

    Raises:
        NotImplementedError: If dtype is not an integer or float ml_dtypes type.
    """
    if _is_float_ml_dtype(dtype):
        fp8_max = float(ml_dtypes.finfo(dtype).max)
        return -fp8_max, fp8_max
    if not compression_types.is_int(dtype):
        raise NotImplementedError(
            "Only support getting quant range for int dtype, "
            f"but got {compression_types.builtin_to_string(dtype)}",
        )
    n_bits = dtype.get_bitwidth()
    signed = not dtype.is_unsigned()
    max_q = 2**n_bits
    if not signed:
        quant_min = 0
        quant_max = max_q - 1
        if mode == QScheme.SYMMETRIC:
            quant_max -= 1
    else:
        quant_min = -max_q / 2
        quant_max = max_q / 2 - 1
        if mode == QScheme.SYMMETRIC:
            quant_min += 1
    return float(quant_min), float(quant_max)


def _quantize_data_by_dtype(
    data: np.ndarray[Any, np.dtype[Any]],
    axes: int | tuple[int, ...],
    dtype: Any,
    quantization_mode: QScheme,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]],
    np.ndarray[Any, np.dtype[Any]] | None,
]:
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Only floating numpy arrays are supported.")

    val_min = np.amin(data, axis=axes, keepdims=True)
    val_max = np.amax(data, axis=axes, keepdims=True)
    epsilon = 1e-6

    if _is_float_ml_dtype(dtype):
        # Float ml_dtypes: always symmetric, zero_point always 0.
        q_val_min, q_val_max = _get_quantize_range_by_dtype(dtype, QScheme.SYMMETRIC)
        max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
        val_min = -max_abs
        val_max = max_abs
        scale = (val_max - val_min + epsilon) / (q_val_max - q_val_min)
        quantized_data = np.clip(data / scale, q_val_min, q_val_max).astype(dtype)
        zero_point: np.ndarray[Any, np.dtype[Any]] | None = np.zeros(
            val_min.shape, dtype=dtype
        ).squeeze()
        scale = scale.astype(data.dtype).squeeze()
        return quantized_data, scale, zero_point

    q_val_min, q_val_max = _get_quantize_range_by_dtype(dtype, quantization_mode)
    zero_point = None

    if quantization_mode == QScheme.SYMMETRIC:
        max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
        val_min = -max_abs
        val_max = max_abs
    else:
        assert quantization_mode == QScheme.ASYMMETRIC
        val_min = np.minimum(0.0, val_min)
        val_max = np.maximum(0.0, val_max)

    scale = (val_max - val_min + epsilon) / (q_val_max - q_val_min)
    quantized_data = np.round(data / scale)

    if quantization_mode == QScheme.SYMMETRIC and dtype.is_unsigned():
        zero_point_shift = q_val_max // 2
        zero_point = zero_point_shift * np.ones(val_min.shape)
    elif quantization_mode == QScheme.ASYMMETRIC:
        zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min + epsilon)
        zero_point = np.round(zero_point)
        zero_point = np.clip(zero_point, q_val_min, q_val_max)

    if zero_point is not None:
        quantized_data += zero_point
        zero_point = zero_point.squeeze()
    quantized_data = np.clip(quantized_data, q_val_min, q_val_max)
    scale = scale.astype(data.dtype).squeeze()

    return quantized_data, scale, zero_point


def _compute_qparams_by_dtype(
    weight: np.ndarray[Any, np.dtype[Any]],
    dtype: Any,
    quantization_mode: QScheme,
    block_sizes: list[int],
) -> (
    tuple[
        np.ndarray[Any, np.dtype[Any]],
        np.ndarray[Any, np.dtype[Any]],
        np.ndarray[Any, np.dtype[Any]] | None,
    ]
    | None
):
    """Compute quantization parameters for a weight array.

    Args:
        weight (np.ndarray): The weight tensor to quantize.
        dtype (Any): The quantization dtype. Either a coreai compression
            builtin int type or an ml_dtypes float numpy dtype.
        quantization_mode (QScheme): One of ``QScheme.SYMMETRIC`` or
            ``QScheme.ASYMMETRIC``.
        block_sizes (list[int]): Block size per axis; ``0`` means no blocking on that
            axis.

    Returns:
        tuple of ``(quantized_data, scale, zero_point)`` or ``None`` if any block size
        is incompatible with the corresponding axis dimension.
    """
    if not isinstance(weight, np.ndarray):
        raise ValueError(
            f"Only numpy arrays are supported, but got weight type {type(weight)}",
        )

    if len(block_sizes) != len(weight.shape):
        raise ValueError(
            "Each axis should have a block size, which means len(block_sizes) must be "
            f"equal to weight's rank, but got {len(block_sizes)} vs {len(weight.shape)}",
        )

    new_shape: list[int] = []
    scale_shape: list[int] = []
    axes_to_skip: list[int] = []
    for axis, (dim_size, block_size) in enumerate(
        zip(weight.shape, block_sizes, strict=False),
    ):
        if block_size > 0:
            if dim_size % block_size != 0:
                logger.warning(
                    "Invalid block_sizes; On %dth axis, the dim size %d is not divisible "
                    "by block size %d. Unable to perform structured quantization.",
                    axis,
                    dim_size,
                    block_size,
                )
                return None
            axes_to_skip.append(len(new_shape))
            num_blocks = dim_size // block_size
            new_shape.extend([num_blocks, block_size])
            scale_shape.append(num_blocks)
        else:
            new_shape.append(dim_size)
            scale_shape.append(1)

    axes = tuple(filter(lambda x: x not in axes_to_skip, range(len(new_shape))))
    quantized_data, scale, zero_point = _quantize_data_by_dtype(
        weight.reshape(new_shape),
        axes,
        dtype,
        quantization_mode,
    )

    np_dtype = (
        compression_types.nptype_from_builtin(dtype) if not _is_float_ml_dtype(dtype) else dtype
    )
    quantized_data = quantized_data.reshape(weight.shape).astype(np_dtype)
    scale = scale.reshape(scale_shape)
    if zero_point is not None:
        zero_point = zero_point.reshape(scale_shape).astype(np_dtype)

    return quantized_data, scale, zero_point
