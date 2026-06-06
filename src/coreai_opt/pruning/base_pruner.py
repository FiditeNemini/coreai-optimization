# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base pruner class."""

from coreai_opt.base_model_compressor import _BaseModelCompressor


class _BasePruner(_BaseModelCompressor):
    """Base class for all pruning compressors."""
