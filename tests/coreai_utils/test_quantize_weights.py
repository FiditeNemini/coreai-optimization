# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import re

import pytest

from coreai_opt.coreai_utils import CompressionGranularity, DType, quantize_weights
from coreai_opt.coreai_utils.common import QScheme
from tests.export.export_utils import MLIRConverter


def _get_bss_scale_element_type(ir: str) -> str | None:
    """Return the element type of the scale operand in the first blockwise_shift_scale op.

    The IR format for the op is:
        coreai.blockwise_shift_scale %data, %scale, %off1, %off2
            : (tensor<AxBxDATA>, tensor<CxDxSCALE>, ...) -> ...
    """
    match = re.search(
        r"coreai\.blockwise_shift_scale[^:]+:\s*\([^,]+,\s*tensor<[^>]*x(\w+)>",
        ir,
    )
    return match.group(1) if match else None


@pytest.mark.parametrize(
    "dtype",
    [
        DType.FP4_E2M1FN,
        DType.FP8_E4M3FN,
        DType.FP8_E5M2,
        DType.INT4,
        DType.INT8,
        DType.UINT4,
        DType.UINT8,
    ],
)
@pytest.mark.parametrize("qscheme", [QScheme.SYMMETRIC, QScheme.ASYMMETRIC])
def test_mlir_weight_quantization(
    dtype: DType,
    qscheme: QScheme,
    _coreai_program,
) -> None:
    """Test MLIR-level weight quantization via coreai_opt.coreai_utils.quantize_weights."""
    coreai_program, _, uncompressed_dtype = _coreai_program

    if not dtype.is_int() and qscheme != QScheme.SYMMETRIC:
        with pytest.raises(ValueError, match="Asymmetric quantization.*is not supported"):
            quantize_weights(
                coreai_program=coreai_program,
                dtype=dtype,
                qscheme=qscheme,
                weight_num_threshold=0,
            )
        return
    compressed = quantize_weights(
        coreai_program=coreai_program,
        dtype=dtype,
        qscheme=qscheme,
        granularity=CompressionGranularity.PER_BLOCK
        if dtype == DType.FP4_E2M1FN
        else CompressionGranularity.PER_TENSOR,
        block_size=32,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.blockwise_shift_scale" in ir

    # Check the quantized dtype appears in the IR.
    if dtype == DType.FP4_E2M1FN:
        assert "f4E2M1FN" in ir
    elif dtype == DType.INT8:
        assert "si8" in ir
    elif dtype == DType.UINT8:
        assert "ui8" in ir
    elif dtype == DType.INT4:
        assert "si4" in ir
    elif dtype == DType.UINT4:
        assert "ui4" in ir
    elif dtype == DType.FP8_E4M3FN:
        assert "f8E4M3FN" in ir
    elif dtype == DType.FP8_E5M2:
        assert "f8E5M2" in ir

    # Uncompressed weight dtype propagates into the scale constants.
    if uncompressed_dtype == "fp16":
        assert "f16" in ir
    else:
        assert "f32" in ir


@pytest.mark.parametrize(
    "dtype, scale_dtype, expected_scale_type",
    [
        # INT dtypes: scale always uses the uncompressed weight dtype.
        (DType.INT4, None, None),
        (DType.INT8, None, None),  # None → checked dynamically from uncompressed_dtype
        # FP4 data dtype: scale is always f8E8M0FNU internally.
        (DType.FP4_E2M1FN, None, "f8E8M0FNU"),
        # FP8 data dtypes with default scale (inherits uncompressed weight dtype).
        (DType.FP8_E4M3FN, None, None),
        (DType.FP8_E5M2, None, None),
        # FP8 data dtypes with explicit FP8_E8M0FNU scale.
        (DType.FP8_E4M3FN, DType.FP8_E8M0FNU, "f8E8M0FNU"),
        (DType.FP8_E5M2, DType.FP8_E8M0FNU, "f8E8M0FNU"),
    ],
)
def test_mlir_weight_quantization_scale_element_type(
    dtype: DType,
    scale_dtype: DType | None,
    expected_scale_type: str | None,
    _coreai_program,
) -> None:
    """Test that the scale operand of blockwise_shift_scale has the expected element type."""
    coreai_program, _, uncompressed_dtype = _coreai_program

    compressed = quantize_weights(
        coreai_program=coreai_program,
        dtype=dtype,
        scale_dtype=scale_dtype,
        granularity=CompressionGranularity.PER_BLOCK
        if dtype == DType.FP4_E2M1FN
        else CompressionGranularity.PER_TENSOR,
        block_size=32,
        weight_num_threshold=0,
        in_place=False,
    )

    scale_elem_type = _get_bss_scale_element_type(str(compressed))
    assert scale_elem_type is not None, "blockwise_shift_scale not found in IR"

    if expected_scale_type is not None:
        assert scale_elem_type == expected_scale_type
    else:
        # scale_dtype=None: scale inherits the uncompressed weight element type.
        assert scale_elem_type == ("f16" if uncompressed_dtype == "fp16" else "f32")


def test_mlir_weight_quantization_weight_num_threshold(_coreai_program) -> None:
    """Test that weights below weight_num_threshold are not compressed."""
    coreai_program, _, _ = _coreai_program

    compressed = quantize_weights(
        coreai_program=coreai_program,
        dtype=DType.INT8,
        weight_num_threshold=int(10e6),
        in_place=False,
    )

    # The linear layer weight (2048 * 32 = 65536 elements) is below 10e6,
    # so no compression should have been applied.
    assert "coreai.blockwise_shift_scale" not in str(compressed)


def test_mlir_weight_quantization_in_place(_exported_program) -> None:
    """Test in_place=False leaves the original program unmodified; in_place=True modifies it."""
    exported_program, _, _ = _exported_program

    # in_place=False: result is a deep copy; original is untouched.
    coreai_program = MLIRConverter._lower_to_coreai(exported_program)
    result = quantize_weights(
        coreai_program=coreai_program,
        dtype=DType.INT8,
        weight_num_threshold=0,
        in_place=False,
    )
    assert result is not coreai_program
    assert "coreai.blockwise_shift_scale" not in str(coreai_program)
    assert "coreai.blockwise_shift_scale" in str(result)

    # in_place=True: result is the same object; original is modified.
    coreai_program = MLIRConverter._lower_to_coreai(exported_program)
    result = quantize_weights(
        coreai_program=coreai_program,
        dtype=DType.INT8,
        weight_num_threshold=0,
        in_place=True,
    )
    assert result is coreai_program
    assert "coreai.blockwise_shift_scale" in str(coreai_program)


@pytest.mark.parametrize(
    "dtype, scale_dtype, error_match",
    [
        # dtype not in _VALID_WEIGHT_DTYPES
        (DType.FP8_E8M0FNU, None, "Unsupported weight dtype"),
        # INT dtype + non-None scale_dtype
        (DType.INT8, DType.FP8_E8M0FNU, "scale_dtype must be None for integer dtype"),
        # FP4 dtype + non-None scale_dtype
        (DType.FP4_E2M1FN, DType.FP8_E8M0FNU, "scale_dtype must be None for FP4 dtype"),
        # FP8 dtype + invalid scale_dtype (INT8 is not a valid scale dtype)
        (DType.FP8_E4M3FN, DType.INT8, "Invalid scale_dtype"),
    ],
)
def test_mlir_weight_quantization_scale_dtype_validation(
    dtype: DType,
    scale_dtype: DType | None,
    error_match: str,
    _coreai_program,
) -> None:
    """Test that invalid dtype/scale_dtype combinations raise ValueError."""
    coreai_program, _, _ = _coreai_program
    with pytest.raises(ValueError, match=error_match):
        quantize_weights(
            coreai_program=coreai_program,
            dtype=dtype,
            scale_dtype=scale_dtype,
            weight_num_threshold=0,
        )


@pytest.mark.parametrize("dtype", [DType.FP8_E4M3FN, DType.FP8_E5M2])
def test_mlir_weight_quantization_scale_dtype_fp8_e8m0fnu(
    dtype: DType,
    _coreai_program,
) -> None:
    """Test that scale_dtype=DType.FP8_E8M0FNU stores the scale in f8E8M0FNU format."""
    coreai_program, _, _ = _coreai_program

    compressed = quantize_weights(
        coreai_program=coreai_program,
        dtype=dtype,
        scale_dtype=DType.FP8_E8M0FNU,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.blockwise_shift_scale" in ir
    assert "f8E8M0FNU" in ir
    if dtype == DType.FP8_E4M3FN:
        assert "f8E4M3FN" in ir
    else:
        assert "f8E5M2" in ir


@pytest.mark.parametrize(
    "granularity, block_size, expected_scale_shape",
    [
        # PER_TENSOR: single scale → scale [1, 1]
        (CompressionGranularity.PER_TENSOR, 32, "1x1"),
        # PER_CHANNEL: one scale per output channel (axis 0 of [32, 2048]) → scale [32, 1]
        (CompressionGranularity.PER_CHANNEL, 32, "32x1"),
        # PER_BLOCK(bs=16): output=32, input blocks=2048/16=128 → scale [32, 128]
        (CompressionGranularity.PER_BLOCK, 16, "32x128"),
    ],
)
def test_mlir_weight_quantization_granularity(
    granularity: CompressionGranularity,
    block_size: int,
    expected_scale_shape: str,
    _coreai_program,
) -> None:
    """Test that the scale tensor shape in the IR reflects the granularity.

    The linear weight is stored as [32, 2048] in the MLIR (original PyTorch weight
    shape, with a separate coreai.transpose op), so output_channel_axis=0 and
    input_channel_axis=1.
    Scale shapes for block_sizes [output_bs, input_bs]:
    - PER_TENSOR: block_sizes=[0,0] → scale [1, 1]
    - PER_CHANNEL: block_sizes=[1,0] → scale [32, 1]
    - PER_BLOCK(block_size=16): block_sizes=[1,16] → scale [32, 128]
    """
    coreai_program, _, _ = _coreai_program

    compressed = quantize_weights(
        coreai_program=coreai_program,
        dtype=DType.INT8,
        granularity=granularity,
        block_size=block_size,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.blockwise_shift_scale" in ir
    assert expected_scale_shape in ir


@pytest.mark.parametrize(
    "granularity, block_size",
    [
        (CompressionGranularity.PER_TENSOR, 32),
        (CompressionGranularity.PER_CHANNEL, 32),
        (CompressionGranularity.PER_BLOCK, 16),
        (CompressionGranularity.PER_BLOCK, 64),
    ],
)
def test_mlir_weight_quantization_fp4_granularity_validation(
    granularity: CompressionGranularity,
    block_size: int,
    _coreai_program,
) -> None:
    """Test that FP4_E2M1FN rejects granularity/block_size combinations that would
    produce an invalid MXFP4 encoding (requires PER_BLOCK + block_size=32)."""
    coreai_program, _, _ = _coreai_program
    with pytest.raises(ValueError, match="DType.FP4_E2M1FN requires"):
        quantize_weights(
            coreai_program=coreai_program,
            dtype=DType.FP4_E2M1FN,
            granularity=granularity,
            block_size=block_size,
            weight_num_threshold=0,
        )


def test_mlir_weight_quantization_fp4(_coreai_program) -> None:
    """Test that FP4_E2M1FN quantization produces f4E2M1FN data with f8E8M0FNU scale."""
    coreai_program, _, _ = _coreai_program

    compressed = quantize_weights(
        coreai_program=coreai_program,
        dtype=DType.FP4_E2M1FN,
        granularity=CompressionGranularity.PER_BLOCK,
        block_size=32,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.blockwise_shift_scale" in ir
    assert "f4E2M1FN" in ir
    assert "f8E8M0FNU" in ir
