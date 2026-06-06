# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Core AI MLIR-level compression passes."""

# Op names whose constant inputs are candidates for compression.
_OPS_WEIGHT_NEED_COMPRESSION = frozenset(
    {
        "coreai.batch_matmul",
        "coreai.conv2d",
        "coreai.decomposable.broadcasting_batch_matmul",
        "coreai.gather_nd",
        "coreai.transpose",
    }
)

__all__ = []
