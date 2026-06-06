# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base factory class for creating compression components from specs."""

from abc import ABC, abstractmethod
from enum import Enum

from coreai_opt._utils.spec_utils import PartialConstructor as _PartialConstructor

from .base import CompressionSpec
from .compression_simulator import CompressionSimulatorBase


class CompressionTargetTensor(Enum):
    """
    Enum to specify the target tensor for compression.

    This is a generic enum that can be used across different compression
    techniques (quantization, palettization, etc.).
    """

    WEIGHT = "weight"
    ACTIVATION = "activation"
    LUT = "lut"


class CompressionComponentFactoryBase(ABC):
    """
    Abstract base class for compression component factories.

    This factory provides a generic interface for creating compression
    components from CompressionSpec instances. Different compression
    techniques (quantization, palettization, etc.) should extend this
    base class to provide their specific implementations.
    """

    @classmethod
    @abstractmethod
    def construct(
        cls, spec: CompressionSpec | None, target: CompressionTargetTensor
    ) -> CompressionSimulatorBase | None:
        """
        Create a compression component instance from a CompressionSpec.

        Args:
            spec: CompressionSpec instance containing configuration
            target: The target tensor for compression (weight or activation)

        Returns:
            CompressionSimulatorBase instance configured from the spec
        """
        pass

    @classmethod
    @abstractmethod
    def construct_partial(
        cls, spec: CompressionSpec | None, target: CompressionTargetTensor
    ) -> _PartialConstructor | None:
        """
        Create a compression component partial object for deferred construction.

        Args:
            spec: CompressionSpec instance containing configuration
            target: The target tensor for compression (weight or activation)

        Returns:
            PartialConstructor: A partial object that can be used for deferred
                          construction of CompressionSimulatorBase instances.
        """
        pass
