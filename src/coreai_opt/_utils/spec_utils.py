# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Generic, TypeVar

from torchao.quantization.pt2e import PartialWrapper

from coreai_opt.config.spec import CompressionSimulatorBase

T = TypeVar("T", bound="CompressionSimulatorBase")


class PartialConstructor(PartialWrapper, Generic[T]):
    """
    coreai-opt wrapper for deferred component construction.

    This class wraps TorchAO's PartialWrapper to provide a coreai-opt-specific
    interface for deferred construction of compression components. It allows
    components to be partially constructed with some arguments, while deferring
    the creation of stateful components (like observers) until instantiation.
    Parameterized by ``T``, the type of CompressionSimulatorBase that will be
    created when this PartialConstructor is called.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """
        Call the partial constructor to create an instance of type T.

        Args:
            *args: Positional arguments to pass to the constructor
            **kwargs: Keyword arguments to pass to the constructor

        Returns:
            An instance of type T (a CompressionSimulatorBase subclass)
        """
        return super().__call__(*args, **kwargs)

    def with_callable_args(self, **kwargs: Callable) -> PartialConstructor[T]:
        """
        Add callable arguments that will be invoked at construction time.

        This is useful when you need to create multiple instances with the same
        constructor arguments, but some arguments should be freshly calculated
        for each instance (e.g., to get different observer instances).

        Args:
            **kwargs: Callable functions that will be invoked at construction time
                     to provide argument values

        Returns:
            A new PartialConstructor with the callable arguments added

        Example:
            >>> def create_observer():
            ...     return MinMaxObserver()
            >>> factory = FakeQuantize.with_args(dtype=torch.int8).with_callable_args(
            ...     observer=create_observer
            ... )
            >>> instance1 = factory()  # observer=create_observer() is called
            >>> instance2 = factory()  # observer=create_observer() is called again
        """
        r = PartialConstructor(p=self.p)
        r.callable_args = {**self.callable_args, **kwargs}
        return r


def with_args(cls: type[T], **kwargs: Any) -> PartialConstructor[T]:
    """
    Create a partial constructor for a class with some arguments pre-filled.

    This allows you to create a factory that can produce multiple instances
    of a class with the same constructor arguments.

    Args:
        cls: The class to create a partial constructor for (must be a
             CompressionSimulatorBase subclass)
        **kwargs: Keyword arguments to pre-fill in the constructor

    Returns:
        A PartialConstructor that will create instances of cls when called

    Example:
        >>> factory = with_args(FakeQuantize, dtype=torch.int8, n_bits=8)
        >>> instance1 = factory()
        >>> instance2 = factory()
    """
    r = PartialConstructor(partial(cls, **kwargs))
    return r
