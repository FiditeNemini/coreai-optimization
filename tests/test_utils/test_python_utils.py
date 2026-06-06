# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

from coreai_opt._utils.python_utils import fqn


class MyClass:
    pass


class Outer:
    class Inner:
        pass


def test_fqn():
    # Test with built-in types
    assert fqn(str) == "builtins.str"
    assert fqn(int) == "builtins.int"
    assert fqn(list) == "builtins.list"

    # Test with a custom class
    assert fqn(MyClass) == f"{__name__}.MyClass"
    assert fqn(Outer) == f"{__name__}.Outer"
    assert fqn(Outer.Inner) == f"{__name__}.Outer.Inner"

    # Test with None
    with pytest.raises(TypeError):
        fqn(None)

    # Test with non-type objects
    with pytest.raises(TypeError):
        fqn(123)
