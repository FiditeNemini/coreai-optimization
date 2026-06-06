# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for lazy import utilities."""

import pytest

from coreai_opt._utils.import_utils import lazy_import_module


class TestLazyImportModule:
    """Tests for the generic lazy_import_module function."""

    def test_successful_import(self):
        """Test that a successful import returns the expected modules."""

        def _import_modules():
            import os  # noqa: PLC0415
            import sys  # noqa: PLC0415

            return sys, os

        sys_result, os_result = lazy_import_module(_import_modules, "Error")

        import os  # noqa: PLC0415
        import sys  # noqa: PLC0415

        assert sys_result is sys
        assert os_result is os

    def test_import_from_syntax(self):
        """Test importing specific items from a module."""

        def _import_from_os():
            from os.path import exists, join  # noqa: F401, PLC0415

            return join, exists

        join_func, exists_func = lazy_import_module(_import_from_os, "Error")

        from os.path import exists, join  # noqa: F401, PLC0415

        assert join_func is join
        assert exists_func is exists

    def test_failed_import(self):
        """Test that ImportError with custom message is raised when import fails."""

        def _import_nonexistent():
            import nonexistent_module_12345  # noqa: F401, PLC0415

            return nonexistent_module_12345

        custom_message = "This is a custom error message for testing"

        with pytest.raises(ImportError) as exc_info:
            lazy_import_module(_import_nonexistent, custom_message)

        assert custom_message in str(exc_info.value)
