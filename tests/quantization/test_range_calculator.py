# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
)
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator


@pytest.fixture(
    params=[
        (0, 10),
        (-10, 0),
        (-10, 10),
        (-20, -10),
        (10, 20),
    ]
)
def range(request):
    return request.param


class TestMinMaxRangeCalculator:
    def _create_tensor_with_range(self, low, high, size=(10, 10)):
        """Creates a tensor of a given size with random values
        ensuring low and high are present
        """
        num_elements = size[0] * size[1]
        rand_vals = torch.randint(low=low, high=high + 1, size=(num_elements - 2,)).to(torch.float)
        x_flat = torch.cat([torch.tensor([low, high], dtype=torch.float), rand_vals])
        return x_flat[torch.randperm(x_flat.shape[0])].reshape(size)

    def test_min_max_range_calculator_per_tensor(self, range):
        low, high = range
        x = self._create_tensor_with_range(low, high)
        obs = MinMaxRangeCalculator(granularity=PerTensorGranularity())
        min_val, max_val = obs(x)
        assert min_val == low and max_val == high

    def test_min_max_range_calculator_per_channel(self, range):
        low, high = range
        x = self._create_tensor_with_range(low, high)
        obs = MinMaxRangeCalculator(granularity=PerChannelGranularity(axis=1))
        min_val, max_val = obs(x)
        assert torch.all(min_val == x.amin(dim=[0], keepdim=False))
        assert torch.all(max_val == x.amax(dim=[0], keepdim=False))

    @pytest.mark.parametrize("block_size", [5, 2])
    def test_min_max_range_calculator_per_block(self, range, block_size):
        low, high = range
        x = self._create_tensor_with_range(low, high)
        x_shape = x.shape
        obs = MinMaxRangeCalculator(granularity=PerBlockGranularity(axis=0, block_size=block_size))
        min_val, max_val = obs(x)
        assert torch.all(
            min_val
            == x.view(x.shape[0] // block_size, block_size, x.shape[1])
            .amin(dim=[1], keepdim=True)
            .view(x_shape[0] // block_size, x.shape[1])
        )
        assert torch.all(
            max_val
            == x.view(x.shape[0] // block_size, block_size, x.shape[1])
            .amax(dim=[1], keepdim=True)
            .view(x_shape[0] // block_size, x.shape[1])
        )

    @pytest.mark.parametrize(
        "granularity",
        [PerChannelGranularity(axis=0), PerBlockGranularity(axis=None, block_size=(1, 1))],
    )
    def test_min_max_range_calculator_no_reduce_needed(self, granularity):
        x = self._create_tensor_with_range(0, 10, size=(2, 1))
        obs = MinMaxRangeCalculator(granularity=granularity)
        min_val, max_val = obs(x)
        assert min_val.shape == (2, 1)
        assert max_val.shape == (2, 1)
