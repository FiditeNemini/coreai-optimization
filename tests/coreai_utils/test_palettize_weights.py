# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coreai_opt.coreai_utils import CompressionGranularity, DType, palettize_weights
from tests.export.export_utils import MLIRConverter


@pytest.mark.parametrize(
    "lut_dtype", [None, DType.FP8_E4M3FN, DType.FP8_E5M2, DType.INT8, DType.UINT8]
)
@pytest.mark.parametrize("n_bits", [2, 4, 6, 8])
def test_mlir_weight_palettization(
    n_bits: int,
    lut_dtype: DType | None,
    _coreai_program,
) -> None:
    """Test MLIR-level weight palettization via coreai_opt.coreai_utils.palettize_weights."""
    coreai_program, _, uncompressed_dtype = _coreai_program

    compressed_coreai_program = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=lut_dtype,
        n_bits=n_bits,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed_coreai_program)

    assert "coreai.lut_to_dense" in ir

    # LUT quantization adds blockwise_shift_scale; plain palettization does not.
    if lut_dtype is not None:
        assert "blockwise_shift_scale" in ir
        if lut_dtype == DType.INT8:
            assert "si8" in ir
        elif lut_dtype == DType.UINT8:
            assert "ui8" in ir
        elif lut_dtype == DType.FP8_E4M3FN:
            assert "f8E4M3FN" in ir
        elif lut_dtype == DType.FP8_E5M2:
            assert "f8E5M2" in ir
    else:
        assert "blockwise_shift_scale" not in ir

    # Index tensor is always ui<n_bits>.
    assert f"ui{n_bits}" in ir

    # Uncompressed weight dtype propagates into the LUT or scale constants.
    if uncompressed_dtype == "fp16":
        assert "f16" in ir
    else:
        assert "f32" in ir


def test_mlir_weight_palettization_weight_num_threshold(_coreai_program) -> None:
    """Test that weights below weight_num_threshold are not compressed."""
    coreai_program, _, _ = _coreai_program

    compressed_coreai_program = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=int(10e6),
        in_place=False,
    )

    # The linear layer weight (2048 * 32 = 65536 elements) is below 10e6,
    # so no compression should have been applied.
    assert "coreai.lut_to_dense" not in str(compressed_coreai_program)


def test_mlir_weight_palettization_num_kmeans_workers(_coreai_program) -> None:
    """Test that palettization results are numerically close regardless of worker count."""
    coreai_program, input_tensor, _ = _coreai_program

    compressed_few_workers = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=0,
        num_kmeans_workers=2,
        in_place=False,
    )
    compressed_many_workers = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=0,
        num_kmeans_workers=4,
        in_place=False,
    )

    converter = MLIRConverter()
    outputs_few_workers = converter._run_inference(compressed_few_workers, input_tensor)
    outputs_many_workers = converter._run_inference(compressed_many_workers, input_tensor)

    assert len(outputs_few_workers) == len(outputs_many_workers)
    for out_few, out_many in zip(outputs_few_workers, outputs_many_workers, strict=True):
        assert torch.allclose(out_few, out_many, atol=1e-4)


def test_mlir_weight_palettization_fast_kmeans_mode(_coreai_program) -> None:
    """Test that fast K-means mode with low rounding precision is close to exact K-means."""
    coreai_program, input_tensor, _ = _coreai_program

    compressed_fast = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=0,
        enable_fast_kmeans_mode=True,
        rounding_precision=4,
        in_place=False,
    )
    compressed_exact = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=0,
        enable_fast_kmeans_mode=False,
        in_place=False,
    )

    converter = MLIRConverter()
    outputs_fast = converter._run_inference(compressed_fast, input_tensor)
    outputs_exact = converter._run_inference(compressed_exact, input_tensor)

    assert len(outputs_fast) == len(outputs_exact)
    for out_fast, out_exact in zip(outputs_fast, outputs_exact, strict=True):
        mse = torch.mean((out_fast - out_exact) ** 2)
        peak = torch.max(torch.abs(out_exact))
        psnr = 10 * torch.log10(peak**2 / mse)
        assert psnr > 20


def test_mlir_weight_palettization_in_place(_exported_program) -> None:
    """Test in_place=False leaves the original program unmodified; in_place=True modifies it."""
    exported_program, _, _ = _exported_program

    # in_place=False: result is a deep copy; original is untouched.
    coreai_program = MLIRConverter._lower_to_coreai(exported_program)
    result = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=0,
        in_place=False,
    )
    assert result is not coreai_program
    assert "coreai.lut_to_dense" not in str(coreai_program)
    assert "coreai.lut_to_dense" in str(result)

    # in_place=True: result is the same object; original is modified.
    coreai_program = MLIRConverter._lower_to_coreai(exported_program)
    result = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=0,
        in_place=True,
    )
    assert result is coreai_program
    assert "coreai.lut_to_dense" in str(coreai_program)


@pytest.mark.parametrize(
    "granularity, group_size, expected_num_groups",
    [
        (CompressionGranularity.PER_TENSOR, 32, 1),
        (CompressionGranularity.PER_CHANNEL, 32, 32),
        (CompressionGranularity.PER_GROUPED_CHANNEL, 16, 2),
    ],
)
def test_mlir_weight_palettization_granularity(
    granularity: CompressionGranularity,
    group_size: int,
    expected_num_groups: int,
    _coreai_program,
) -> None:
    """Test that the LUT shape in the IR reflects the granularity.

    The LUT tensor shape is num_groups × 1 × 2^n_bits × 1:
    - PER_TENSOR: num_groups=1 → 1×1×16×1
    - PER_CHANNEL: num_groups=32 (one per output channel) → 32×1×16×1
    - PER_GROUPED_CHANNEL (group_size=16): num_groups=2 → 2×1×16×1
    """
    coreai_program, _, _ = _coreai_program

    n_bits = 4
    compressed = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        n_bits=n_bits,
        granularity=granularity,
        group_size=group_size,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.lut_to_dense" in ir
    assert f"{expected_num_groups}x1x{2**n_bits}x1x" in ir


def test_mlir_weight_palettization_cluster_dim(_coreai_program) -> None:
    """Test that cluster_dim=2 produces a LUT with vector centroids of size cluster_dim.

    For scalar palettization (cluster_dim=1), the LUT shape is 1×num_groups×2^n_bits×1.
    For 2-D palettization (cluster_dim=2), each centroid is a 2-element vector, so the
    LUT shape becomes 1×num_groups×2^n_bits×cluster_dim (e.g., 1×1×16×2 for n_bits=4).
    """
    coreai_program, _, _ = _coreai_program

    n_bits = 4
    cluster_dim = 2
    compressed = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        n_bits=n_bits,
        cluster_dim=cluster_dim,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.lut_to_dense" in ir
    # LUT last dimension equals cluster_dim (vector centroid size).
    assert f"1x1x{2**n_bits}x{cluster_dim}x" in ir


@pytest.mark.parametrize(
    "enable_per_channel_scale, expect_blockwise_shift_scale",
    [
        (True, True),
        (False, False),
    ],
)
def test_mlir_weight_palettization_per_channel_scale(
    enable_per_channel_scale: bool,
    expect_blockwise_shift_scale: bool,
    _coreai_program,
) -> None:
    """Test that enable_per_channel_scale controls presence of blockwise_shift_scale.

    With lut_dtype=None, per-channel scaling is the only source of blockwise_shift_scale.
    """
    coreai_program, _, _ = _coreai_program

    compressed = palettize_weights(
        coreai_program=coreai_program,
        lut_dtype=None,
        weight_num_threshold=0,
        enable_per_channel_scale=enable_per_channel_scale,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.lut_to_dense" in ir
    if expect_blockwise_shift_scale:
        assert "blockwise_shift_scale" in ir
    else:
        assert "blockwise_shift_scale" not in ir
