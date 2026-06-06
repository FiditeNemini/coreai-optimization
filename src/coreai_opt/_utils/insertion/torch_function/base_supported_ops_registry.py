# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Abstract base class for compressor-specific operation registries."""

from abc import ABC
from collections.abc import Callable

from coreai_opt._utils.registry_utils import ClassRegistryMixin


class BaseSupportedOpsRegistry(ClassRegistryMixin, ABC):
    """Abstract base class for compressor-specific operation registries.

    This class provides common functionality for all compressor registries:
    - Getting all supported operations
    - Checking if an operation is supported
    - Getting registry entry classes for functions

    Compressor-specific registries should inherit from this class and register
    their supported operations using the @register_class decorator.
    """

    @classmethod
    def get_supported_ops(cls) -> tuple[Callable, ...]:
        """Return tuple of all registered operations for this compressor."""
        ops = []
        for entry_class in cls.REGISTRY.values():
            if hasattr(entry_class, "ops") and entry_class.ops:
                ops.extend(entry_class.ops)
        return tuple(ops)

    @classmethod
    def supports_operation(cls, func: Callable) -> bool:
        """Check if operation is supported by this compressor."""
        return func in cls.get_supported_ops()

    @classmethod
    def get_registry_entry_for_func(cls, func: Callable) -> type | None:
        """Get registry entry class for the given function.

        Args:
            func: The function to get registry entry for

        Returns:
            The registry entry class, or None if not found
        """
        for entry_class in cls.REGISTRY.values():
            if hasattr(entry_class, "ops") and func in entry_class.ops:
                return entry_class
        return None

    @classmethod
    def get_func_type(cls, func: Callable) -> str | None:
        """Get func type for the function using the registered registry class name.

        Args:
            func: The function to get the type for

        Returns:
            The function type, or None if not found
        """
        for func_type, entry_class in cls.REGISTRY.items():
            if hasattr(entry_class, "ops") and entry_class.ops and func in entry_class.ops:
                return func_type
        return None
