# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for KMeans support mixins that handle operation-specific reshaping."""

import pytest
import torch

from coreai_opt.palettization.kmeans.kmeans_support_mixins import (
    _ConvPalettizationMixin,
    _LinearPalettizationMixin,
    _PalettizationSupportMixin,
)


class TestLinearPalettizationMixin:
    """Test _LinearPalettizationMixin for linear layer weight handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mixin = _LinearPalettizationMixin()

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1),
            (10, 5),
            (100, 50),
            (256, 768),
            (8, 4),
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_reshape_for_kmeans(self, shape, axis):
        """Test that _LinearPalettizationMixin reshape is a no-op for 2D tensors."""
        weight = torch.randn(shape)
        reshaped = self.mixin.reshape_for_kmeans(weight, axis=axis)
        assert torch.equal(reshaped, weight)  # Should be no-op
        assert reshaped.shape == shape

    @pytest.mark.parametrize(
        "shape",
        [
            (10, 5),
            (8, 4),
            (100, 200),
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_reshape_to_original(self, shape, axis):
        """Test that _LinearPalettizationMixin reshape_to_original is a no-op."""
        clustered_weight = torch.randn(shape)
        result = self.mixin.reshape_to_original(clustered_weight, axis=axis, original_shape=shape)
        assert torch.equal(result, clustered_weight)  # Should be no-op
        assert result.shape == shape

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1),
            (8, 4),
            (100, 50),
            (256, 768),
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_round_trip_consistency(self, shape, axis):
        """Test that reshape_for_kmeans + reshape_to_original is identity."""
        weight = torch.randn(shape)
        original_shape = weight.shape

        # Forward reshape
        reshaped = self.mixin.reshape_for_kmeans(weight, axis=axis)

        # Backward reshape
        restored = self.mixin.reshape_to_original(
            reshaped, axis=axis, original_shape=original_shape
        )

        # Should be identical
        assert torch.equal(restored, weight)
        assert restored.shape == original_shape


class TestConvPalettizationMixin:
    """Test _ConvPalettizationMixin for convolutional layer weight handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mixin = _ConvPalettizationMixin()

    @pytest.mark.parametrize(
        "shape,expected_axis0_shape,expected_axis1_shape",
        [
            # Conv2d shapes: [out_channels, in_channels, kernel_h, kernel_w]
            ((32, 16, 3, 3), (32, 144), (288, 16)),  # Standard Conv2d
            ((128, 64, 1, 1), (128, 64), (128, 64)),  # 1x1 Conv2d
            ((64, 32, 5, 5), (64, 800), (1600, 32)),  # Large kernel Conv2d
            ((1, 3, 7, 7), (1, 147), (49, 3)),  # Single output channel
            # Conv1d shapes: [out_channels, in_channels, kernel_size]
            ((64, 32, 5), (64, 160), (320, 32)),  # Conv1d
            ((128, 256, 3), (128, 768), (384, 256)),  # Conv1d large
            # Conv3d shapes: [out_channels, in_channels, d, h, w]
            ((32, 16, 2, 3, 3), (32, 288), (576, 16)),  # Conv3d
            ((8, 4, 3, 2, 2), (8, 48), (96, 4)),  # Conv3d small
        ],
    )
    def test_reshape_for_kmeans_shapes(self, shape, expected_axis0_shape, expected_axis1_shape):
        """Test reshape_for_kmeans produces correct output shapes."""
        weight = torch.randn(shape)

        # Test axis=0 (output channels)
        reshaped_0 = self.mixin.reshape_for_kmeans(weight, axis=0)
        assert reshaped_0.shape == expected_axis0_shape

        # Test axis=1 (input channels)
        reshaped_1 = self.mixin.reshape_for_kmeans(weight, axis=1)
        assert reshaped_1.shape == expected_axis1_shape

    @pytest.mark.parametrize(
        "shape",
        [
            (32, 16, 3, 3),  # Conv2d
            (64, 32, 5),  # Conv1d
            (32, 16, 2, 3, 3),  # Conv3d
            (1, 1, 3, 3),  # Single channel Conv2d
            (128, 64, 1, 1),  # 1x1 Conv2d
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_round_trip_consistency(self, shape, axis):
        """Test round-trip consistency for various conv shapes and axes."""
        weight = torch.randn(shape)
        original_shape = weight.shape

        # Forward reshape
        reshaped = self.mixin.reshape_for_kmeans(weight, axis=axis)

        # Backward reshape
        restored = self.mixin.reshape_to_original(
            reshaped, axis=axis, original_shape=original_shape
        )

        # Should be identical
        assert torch.equal(restored, weight)
        assert restored.shape == original_shape


class TestDefaultAxisEnforcement:
    """Concrete subclasses of _PalettizationSupportMixin must define default_axis."""

    def test_subclass_without_default_axis_raises(self):
        with pytest.raises(TypeError, match="default_axis"):

            class _MissingDefaultAxis(_PalettizationSupportMixin):
                def reshape_for_kmeans(self, weight, axis):
                    return weight

                def reshape_to_original(self, clustered_weight, axis, original_shape):
                    return clustered_weight

    @pytest.mark.parametrize("invalid_axis", [-1, 2, 3])
    def test_subclass_with_invalid_default_axis_raises(self, invalid_axis):
        with pytest.raises(ValueError, match="must be 0 or 1"):

            class _InvalidDefaultAxis(_PalettizationSupportMixin):
                default_axis = invalid_axis

                def reshape_for_kmeans(self, weight, axis):
                    return weight

                def reshape_to_original(self, clustered_weight, axis, original_shape):
                    return clustered_weight
