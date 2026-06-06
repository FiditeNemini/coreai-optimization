# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import torch
import torch.nn as nn

from coreai_opt.base_model_compressor import (
    _COREAI_OPT_PREPARED_ATTR,
    _BaseModelCompressor,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestIsModelPrepared:
    """Tests for _is_model_prepared static method."""

    def test_fresh_model_not_prepared(self):
        """Test that a newly created model is not marked as prepared."""
        model = SimpleModel()
        assert not _BaseModelCompressor._is_model_prepared(model)


class TestMarkModelAsPrepared:
    """Tests for _mark_model_as_prepared static method."""

    def test_mark_simple_model_as_prepared(self):
        """Test marking a simple model as prepared."""
        model = SimpleModel()
        assert not _BaseModelCompressor._is_model_prepared(model)

        _BaseModelCompressor._mark_model_as_prepared(model)

        assert _BaseModelCompressor._is_model_prepared(model)
        # Marker is registered as a non-persistent buffer.
        assert _COREAI_OPT_PREPARED_ATTR in dict(model.named_buffers())
        assert _COREAI_OPT_PREPARED_ATTR not in model.state_dict()

    def test_mark_already_prepared_model(self):
        """Test that marking an already prepared model is idempotent."""
        model = SimpleModel()
        _BaseModelCompressor._mark_model_as_prepared(model)
        assert _BaseModelCompressor._is_model_prepared(model)

        # Mark again
        _BaseModelCompressor._mark_model_as_prepared(model)
        assert _BaseModelCompressor._is_model_prepared(model)
        assert _COREAI_OPT_PREPARED_ATTR in dict(model.named_buffers())


class TestPreparedSurvivesDeepcopy:
    """The marker must survive deepcopy across module types used by coreai-opt.

    ``torch.fx.GraphModule.__deepcopy__`` only copies parameters, buffers, and
    submodules; arbitrary attributes set via ``setattr`` are dropped. Storing the
    marker as a buffer ensures PT2E-prepared models stay marked after deepcopy.
    """

    def test_nn_module_deepcopy(self):
        model = SimpleModel()
        _BaseModelCompressor._mark_model_as_prepared(model)
        assert _BaseModelCompressor._is_model_prepared(copy.deepcopy(model))

    def test_exported_graph_module_deepcopy(self):
        exported = torch.export.export(SimpleModel(), (torch.randn(2, 10),))
        graph_module = exported.module()
        _BaseModelCompressor._mark_model_as_prepared(graph_module)
        assert _BaseModelCompressor._is_model_prepared(copy.deepcopy(graph_module))
