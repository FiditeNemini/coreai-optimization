# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


@pytest.fixture
def simple_conv_linear_model():
    """Fixture providing a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_model_input():
    """Fixture providing example input tensor."""
    return torch.randn(1, 1, 28, 28)


class SharedParamsModel(nn.Module):
    """Simple model with shared parameters across two layers."""

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        # Create a single linear layer whose parameters will be shared
        self.shared_linear = nn.Linear(hidden_size, hidden_size)

        # Create input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Create two separate layers that will share the same parameters
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

        # Share the parameters between layer1 and layer2
        self.layer1.weight = self.shared_linear.weight
        self.layer1.bias = self.shared_linear.bias
        self.layer2.weight = self.shared_linear.weight
        self.layer2.bias = self.shared_linear.bias

        # Output layer
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Apply input layer
        x = self.relu(self.input_layer(x))

        # Apply first shared layer
        x1 = self.relu(self.layer1(x))

        # Apply second shared layer (with same parameters)
        x2 = self.relu(self.layer2(x1))

        # Output layer
        output = self.output(x2)
        return output


@pytest.fixture
def shared_params_model():
    """Fixture providing a model with shared parameters for testing."""
    return SharedParamsModel()


@pytest.fixture
def shared_params_model_input():
    """Fixture providing example input tensor for shared params model."""
    return torch.randn(1, 784)


class GatedMLPModel(nn.Module):
    """Simple gated MLP model with uniform activation tensor rank.
    All intermediate activations are rank-3 tensors (batch, seq_len, dim or
    hidden_dim), making this model suitable for testing per-channel activation
    quantization across a wide range of axis values (0, 1, 2, -1, -2, -3)
    without out-of-bounds errors.
    Inspired by the MLP block in Qwen3 and similar transformer architectures.
    """

    def __init__(self, dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_tensor = self.up_proj(x)
        gate_tensor = nn.functional.silu(self.gate_proj(x))
        return self.down_proj(up_tensor * gate_tensor)


@pytest.fixture
def gated_mlp_model():
    """Fixture providing a gated MLP model with uniform activation rank."""
    return GatedMLPModel()


@pytest.fixture
def gated_mlp_model_input():
    """Fixture providing example input tensor for gated MLP model."""
    return torch.randn(1, 4, 32)


class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


@pytest.fixture
def simple_linear_model():
    """Fixture providing a minimal linear model for testing."""
    return SimpleLinearModel()


@pytest.fixture
def simple_linear_model_input():
    """Fixture providing example input tensor."""
    return torch.randn(4, 64)


class SimpleMHAModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.linear(attn_out)


@pytest.fixture
def simple_mha_model():
    """Fixture providing a model with MultiheadAttention for testing."""
    return SimpleMHAModel()


@pytest.fixture
def simple_mha_model_input():
    """Fixture providing example input tensor for MHA model."""
    return torch.randn(1, 10, 64)
