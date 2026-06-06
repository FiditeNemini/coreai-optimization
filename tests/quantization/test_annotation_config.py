# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for PT2E AnnotationConfig conversion."""

import pytest
import torch
from torchao.quantization.pt2e.quantizer import (
    QuantizationSpec as TorchAOQuantizationSpec,
)

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization._graph._annotation_config import AnnotationConfig
from coreai_opt.quantization.spec import QuantizationSpec
from coreai_opt.quantization.spec.granularity import PerTensorGranularity


@pytest.mark.parametrize(
    "dtype,qscheme,expected_quant_min,expected_quant_max",
    [
        (torch.int8, "symmetric", -128, 127),
        (torch.int8, "asymmetric", -128, 127),
        (torch.uint8, "symmetric", 0, 255),
        (torch.float8_e4m3fn, "symmetric", -448.0, 448.0),
        (torch.float8_e5m2, "symmetric", -57344.0, 57344.0),
        (torch.float4_e2m1fn_x2, "symmetric", -6.0, 6.0),
    ],
)
def test_convert_to_pt2e_spec_preserves_attributes(
    dtype, qscheme, expected_quant_min, expected_quant_max
):
    """
    Verify coreai_opt QuantizationSpec attributes are preserved when converting to
    TorchAO QuantizationSpec. Regression test for quant_min/quant_max override bug.
    """
    coreai_opt_spec = QuantizationSpec(
        dtype=dtype,
        qscheme=qscheme,
        granularity=PerTensorGranularity(),
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    torchao_spec = AnnotationConfig._convert_to_pt2e_spec(
        coreai_opt_spec, CompressionTargetTensor.WEIGHT
    )

    assert isinstance(torchao_spec, TorchAOQuantizationSpec)
    assert torchao_spec.dtype == dtype
    assert torchao_spec.quant_min == expected_quant_min
    assert torchao_spec.quant_max == expected_quant_max
    assert torchao_spec.observer_or_fake_quant_ctr is not None
    assert callable(torchao_spec.observer_or_fake_quant_ctr)
