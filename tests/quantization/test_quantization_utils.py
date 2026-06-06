# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest
import torch

from coreai_opt.quantization.spec import QuantizationFormulation
from coreai_opt.quantization.spec.fake_quantize import (
    _dequantize_float,
    _dequantize_int,
    _quantize_float,
    _quantize_int,
)


@pytest.mark.parametrize(
    "qformulation", [QuantizationFormulation.ZP, QuantizationFormulation.MINVAL]
)
@pytest.mark.parametrize(
    "minval, quant_min, quant_max",
    [(-12.75, -128, 127), (-0.3, 0, 255)],
    ids=["symmetric", "asymmetric"],
)
def test_quantize_dequantize_int_roundtrip(qformulation, minval, quant_min, quant_max):
    """Roundtrip quantize_int -> dequantize_int has at most scale/2 + eps error per element."""
    tensor = torch.tensor([0.0, 0.1, 0.25, 0.5, 1.0, -0.3])
    scale = torch.tensor(0.1)

    if qformulation == QuantizationFormulation.MINVAL:
        quant_offset = torch.tensor(quant_min)
        float_offset = torch.tensor(minval)
    elif qformulation == QuantizationFormulation.ZP:
        zero_point = np.clip(round(quant_min - minval / scale.item()), quant_min, quant_max).item()
        quant_offset = torch.tensor(zero_point)
        float_offset = torch.tensor(0.0)
    else:
        raise NotImplementedError(qformulation)

    quantized, _ = _quantize_int(tensor, scale, quant_offset, float_offset, quant_min, quant_max)
    reconstructed = _dequantize_int(quantized, scale, quant_offset, float_offset)

    torch.testing.assert_close(reconstructed, tensor, atol=scale / 2 + 1e-5, rtol=0)


@pytest.mark.parametrize(
    "qformulation",
    [QuantizationFormulation.ZP, QuantizationFormulation.MINVAL],
)
def test_quantize_int_clamps_and_mask(qformulation):
    """quantize_int clamps to [quant_min, quant_max] and returns correct mask."""
    quant_min, quant_max = -2, 2
    scale = torch.tensor(1.0)

    # Pre-clamp continuous q values: [-5, -2, 0, 2, 5]
    # (identical under both ZP and MINVAL formulations).
    tensor = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])

    if qformulation == QuantizationFormulation.ZP:
        quant_offset = torch.tensor(0)
        float_offset = torch.tensor(0)
    elif qformulation == QuantizationFormulation.MINVAL:
        quant_offset = torch.tensor(quant_min)
        float_offset = scale * quant_min
    else:
        raise NotImplementedError(qformulation)

    quantized, mask = _quantize_int(tensor, scale, quant_offset, float_offset, quant_min, quant_max)

    assert quantized.min() >= quant_min
    assert quantized.max() <= quant_max
    expected_mask = torch.tensor([False, True, True, True, False])
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize(
    "dtype, atol",
    [
        # e4m3fn: 3 mantissa bits, max test value 2.0 -> ULP = 2^(1-3) = 0.25,
        # so max rounding error = ULP/2 = 0.125
        (torch.float8_e4m3fn, 0.125),
        # e5m2: 2 mantissa bits, max test value 2.0 -> ULP = 2^(1-2) = 0.5,
        # so max rounding error = ULP/2 = 0.25
        (torch.float8_e5m2, 0.25),
    ],
    ids=["e4m3fn", "e5m2"],
)
def test_quantize_dequantize_float_roundtrip(dtype, atol):
    """Roundtrip quantize_float -> dequantize_float is close to original for fp8."""
    tensor = torch.tensor([0.1, 0.5, 1.0, 2.0, -0.5, -1.0])
    scale = torch.tensor(1.0)
    finfo = torch.finfo(dtype)
    quant_min = finfo.min
    quant_max = finfo.max

    quantized, _ = _quantize_float(tensor, scale, quant_min, quant_max, dtype)
    reconstructed = _dequantize_float(quantized, scale)

    torch.testing.assert_close(reconstructed, tensor, atol=atol, rtol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ],
    ids=["e4m3fn", "e5m2"],
)
def test_quantize_float_clamps_and_mask(dtype):
    """quantize_float mask is True for in-range values, False for clipped."""
    finfo = torch.finfo(dtype)
    quant_min = finfo.min
    quant_max = finfo.max
    scale = torch.tensor(1.0)

    # Values: well inside range, at max boundary, beyond max, at min boundary, beyond min
    in_range_val = quant_max / 2.0
    tensor = torch.tensor(
        [in_range_val, quant_max, quant_max + 100.0, quant_min, quant_min - 100.0]
    )

    _, mask = _quantize_float(tensor, scale, quant_min, quant_max, dtype)

    expected_mask = torch.tensor([True, True, False, True, False])
    assert torch.equal(mask, expected_mask)


def test_quantize_float_invalid_dtype():
    """quantize_float raises ValueError for non-float4/float8 dtype."""
    tensor = torch.tensor([1.0, 2.0])
    scale = torch.tensor(1.0)

    with pytest.raises(ValueError, match="Expected float4/float8 dtype, got"):
        _quantize_float(tensor, scale, -1.0, 1.0, torch.float16)


def test_quantize_float_fp4():
    """quantize_float roundtrip and mask work for fp4 dtype."""
    tensor = torch.tensor([0.5, 1.0, 2.0, -0.5, -1.0])
    scale = torch.tensor(1.0)
    quant_min = -6.0
    quant_max = 6.0

    quantized, mask = _quantize_float(tensor, scale, quant_min, quant_max, torch.float4_e2m1fn_x2)
    reconstructed = _dequantize_float(quantized, scale)

    # All values are within range, so mask should be all True
    assert mask.all()

    # fp4 e2m1: 1 mantissa bit, max test value 2.0 -> ULP = 2^(1-1) = 1.0,
    # so max rounding error = ULP/2 = 0.5
    torch.testing.assert_close(reconstructed, tensor, atol=0.5, rtol=0)
