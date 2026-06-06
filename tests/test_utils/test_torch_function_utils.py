# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for insertion utilities."""

from collections import deque

import pytest
import torch

from coreai_opt._utils.insertion.torch_function.module_boundary_tracker import (
    ModuleBoundaryInfo,
    ModuleBoundaryTracker,
    TensorIdVersion,
)
from coreai_opt._utils.torch_utils import NamedModule


@pytest.mark.parametrize("in_place", [False, True])
def test_module_boundary_tracker(simple_conv_linear_model, simple_model_input, in_place):
    module_boundary_tracker = ModuleBoundaryTracker()

    class ReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp):
            return torch.relu_(inp)

    if in_place:
        simple_conv_linear_model.relu = ReLU()

    relu_input_id_version = None
    relu_output_id_version = None

    def forward_pre_hook(self, inps):
        nonlocal relu_input_id_version
        module_boundary_tracker.record_module_boundary_tensors(
            (type(self).__name__, self), inps, "input"
        )
        if type(self).__name__ == "ReLU":
            relu_input_id_version = TensorIdVersion(id(inps[0]), inps[0]._version)

    def forward_hook(self, _, out):
        nonlocal relu_output_id_version
        module_boundary_tracker.record_module_boundary_tensors(
            (type(self).__name__, self), out, "output"
        )
        if type(self).__name__ == "ReLU":
            relu_output_id_version = TensorIdVersion(id(out), out._version)

    for module in simple_conv_linear_model.modules():
        module.register_forward_hook(forward_hook)
        module.register_forward_pre_hook(forward_pre_hook)

    _ = simple_conv_linear_model(simple_model_input)

    # Check that in_place operation actually took place
    if in_place:
        assert relu_input_id_version.id == relu_output_id_version.id
        assert relu_input_id_version.version == 0
        assert relu_output_id_version.version == 1
    else:
        assert relu_input_id_version.id != relu_output_id_version.id
        assert relu_input_id_version.version == 0
        assert relu_output_id_version.version == 0

    # Regardless of in_place or not, the module boundary tensors should end up being tracked
    # the same. Keys are the counters which uniquely identify the tensor
    # objects (solving GC reuse of id()) and version distinguishes in-place mutation states.
    expected_values = [
        {
            "input": deque(
                [
                    ModuleBoundaryInfo(NamedModule("Conv2d", simple_conv_linear_model.conv), 0),
                    ModuleBoundaryInfo(NamedModule("SimpleModel", simple_conv_linear_model), 0),
                ]
            ),
            "output": deque(),
        },
        {
            "input": deque(
                [ModuleBoundaryInfo(NamedModule("ReLU", simple_conv_linear_model.relu), 0)]
            ),
            "output": deque(
                [ModuleBoundaryInfo(NamedModule("Conv2d", simple_conv_linear_model.conv), 0)]
            ),
        },
        {
            "input": deque(),
            "output": deque(
                [ModuleBoundaryInfo(NamedModule("ReLU", simple_conv_linear_model.relu), 0)]
            ),
        },
        {
            "input": deque(
                [ModuleBoundaryInfo(NamedModule("Linear", simple_conv_linear_model.linear), 0)]
            ),
            "output": deque(),
        },
        {
            "input": deque(),
            "output": deque(
                [
                    ModuleBoundaryInfo(NamedModule("Linear", simple_conv_linear_model.linear), 0),
                    ModuleBoundaryInfo(NamedModule("SimpleModel", simple_conv_linear_model), 0),
                ]
            ),
        },
    ]

    for tensor_boundary in module_boundary_tracker.tensor_boundaries.values():
        assert tensor_boundary in expected_values

    assert len(module_boundary_tracker.tensor_boundaries) == len(expected_values)
