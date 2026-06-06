# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


class _IncompatibleGranularityError(Exception):
    """Raised when tensor is incompatible with granularity requirements."""

    pass


class _IncompatibleClusterDimError(Exception):
    """Raised when tensor dimensions are incompatible with cluster_dim."""

    pass
