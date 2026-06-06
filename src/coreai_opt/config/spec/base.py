# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base classes for compression specifications.

This module defines the common base class and enums used by all compression
techniques (quantization, palettization, pruning, etc.).
"""

from pydantic import BaseModel, ConfigDict

from coreai_opt.common import CompressionType


class CompressionSpec(BaseModel):
    """
    Base class for compression specifications.

    This class provides common infrastructure for all compression techniques
    including quantization, palettization, pruning, and distillation.

    All concrete compression specs (QuantizationSpec, PalettizationSpec, etc.)
    should inherit from this base class and set the `_compression_type` private
    attribute to identify their compression type.

    Attributes:
        model_config: Pydantic configuration making specs immutable (frozen=True)
            and rejecting extra fields (extra="forbid")
        _compression_type: Private attribute that must be set by subclasses to
            identify the compression type
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    def get_compression_type(self) -> CompressionType:
        """
        Return the type of compression this spec represents.

        This method reads from the `_compression_type` private attribute that
        must be set by each concrete subclass.

        Returns:
            CompressionType enum value

        Example:
            >>> spec = QuantizationSpec(...)
            >>> spec.get_compression_type()
            CompressionType.QUANTIZATION
        """
        return self._compression_type
