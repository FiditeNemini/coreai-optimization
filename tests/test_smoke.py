# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Smoke tests for coreai_opt package."""

import pytest
import torch

import coreai_opt
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    default_weight_palettization_spec,
)
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.spec import (
    default_weight_quantization_spec,
)


class TestSmokeTest:
    """Basic smoke tests to verify package functionality."""

    def test_coreai_opt_import(self):
        """Test that coreai_opt can be imported."""
        assert coreai_opt.__version__ is not None

    @pytest.mark.parametrize("execution_mode", ["eager", "graph"])
    def test_basic_quantization(self, execution_mode):
        """Test basic quantization functionality."""
        # Create a simple model
        model = torch.nn.Linear(10, 5)

        # Create quantizer config with default spec
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec=None,
                op_output_spec=None,
            ),
            execution_mode=execution_mode,
        )

        # Create and prepare quantizer
        quantizer = Quantizer(model, config)
        sample_input = torch.randn(1, 10)
        quantizer.prepare(example_inputs=(sample_input,))

    def test_basic_palettization(self):
        """Test basic palettization functionality."""
        # Create a simple model
        model = torch.nn.Linear(10, 5)

        # Create palettizer config with default spec
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": default_weight_palettization_spec()}
            )
        )

        # Create and prepare palettizer
        palettizer = KMeansPalettizer(model, config)
        sample_input = torch.randn(1, 10)
        palettizer.prepare((sample_input,))
