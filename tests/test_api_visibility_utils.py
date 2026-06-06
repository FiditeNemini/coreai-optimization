# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for coreai_opt._utils.api_visibility_utils.

Each test builds a minimal temporary package structure under tmp_path, injects it into sys.path,
imports the relevant modules, and verifies the utility functions behave as documented. Cleanup
removes the injected path entry and any cached module entries from sys.modules.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

from coreai_opt._utils.api_visibility_utils import (
    accessible_public_names,
    collect_declared_obj_ids,
    find_names_missing_from_all,
    find_public_modules,
    find_public_packages,
    originating_public_names,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _make_package(root: Path, rel_path: str, content: str = "") -> Path:
    """Create a Python package or module file under root.

    If rel_path ends with __init__.py, all parent directories are also initialized as packages.
    """
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    # Ensure every parent directory is a package.
    for parent in target.parents:
        if parent == root:
            break
        init = parent / "__init__.py"
        if not init.exists():
            init.write_text("")
    target.write_text(content)
    return target


@pytest.fixture
def tmp_pkg(tmp_path: Path) -> tuple[Path, str]:
    """Create a minimal temporary root package and inject tmp_path into sys.path.

    Yields (tmp_path, root_package_name) and cleans up on teardown.
    """
    pkg_name = "tmp_test_pkg"
    _make_package(tmp_path, f"{pkg_name}/__init__.py")

    sys.path.insert(0, str(tmp_path))
    yield tmp_path, pkg_name

    sys.path.remove(str(tmp_path))
    to_remove = [k for k in sys.modules if k == pkg_name or k.startswith(f"{pkg_name}.")]
    for key in to_remove:
        del sys.modules[key]


# --------------------------------------------------------------------------------------
# find_public_packages
# --------------------------------------------------------------------------------------


class TestFindPublicPackages:
    """Tests for find_public_packages."""

    def test_includes_root(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_packages includes the root package itself."""
        _tmp_path, pkg_name = tmp_pkg
        result = find_public_packages(pkg_name)
        assert pkg_name in result

    def test_includes_public_subpackage(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_packages includes sub-packages with no '_' in their path."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/sub/__init__.py")
        result = find_public_packages(pkg_name)
        assert f"{pkg_name}.sub" in result

    def test_excludes_private_subpackage(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_packages excludes packages whose name starts with '_'."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/_private/__init__.py")
        result = find_public_packages(pkg_name)
        assert f"{pkg_name}._private" not in result

    def test_excludes_nested_under_private(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_packages excludes packages nested under a private package."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/_private/nested/__init__.py")
        result = find_public_packages(pkg_name)
        assert f"{pkg_name}._private.nested" not in result


# --------------------------------------------------------------------------------------
# find_public_modules
# --------------------------------------------------------------------------------------


class TestFindPublicModules:
    """Tests for find_public_modules."""

    def test_includes_public_module(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_modules includes .py files with no '_' in their path."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/public_mod.py")
        assert f"{pkg_name}.public_mod" in find_public_modules(pkg_name)

    def test_excludes_private_module(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_modules excludes modules whose name starts with '_'."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/_private_mod.py")
        assert f"{pkg_name}._private_mod" not in find_public_modules(pkg_name)

    def test_excludes_init_files(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_modules excludes __init__.py files (packages, not modules)."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/sub/__init__.py")
        result = find_public_modules(pkg_name)
        assert f"{pkg_name}.sub" not in result
        assert f"{pkg_name}.__init__" not in result

    def test_excludes_module_under_private_package(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_public_modules excludes modules nested under a private package."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/_private/mod.py")
        assert f"{pkg_name}._private.mod" not in find_public_modules(pkg_name)


# --------------------------------------------------------------------------------------
# originating_public_names
# --------------------------------------------------------------------------------------


class TestOriginatingPublicNames:
    """Tests for originating_public_names."""

    def test_returns_defined_class(self, tmp_pkg: tuple[Path, str]) -> None:
        """originating_public_names returns classes defined in the module."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/mod.py", "class MyClass:\n    pass\n")
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        assert "MyClass" in originating_public_names(mod)

    def test_returns_defined_function(self, tmp_pkg: tuple[Path, str]) -> None:
        """originating_public_names returns functions defined in the module."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/mod.py", "def my_func():\n    pass\n")
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        assert "my_func" in originating_public_names(mod)

    def test_excludes_imported_class(self, tmp_pkg: tuple[Path, str]) -> None:
        """originating_public_names excludes names imported from another module."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/other.py", "class Other:\n    pass\n")
        _make_package(tmp_path, f"{pkg_name}/mod.py", f"from {pkg_name}.other import Other\n")
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        assert "Other" not in originating_public_names(mod)

    def test_excludes_private_names(self, tmp_pkg: tuple[Path, str]) -> None:
        """originating_public_names excludes names starting with '_'."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/mod.py", "_PRIVATE = 1\nPUBLIC = 2\n")
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        names = originating_public_names(mod)
        assert "_PRIVATE" not in names
        assert "PUBLIC" in names

    def test_excludes_submodules(self, tmp_pkg: tuple[Path, str]) -> None:
        """originating_public_names excludes submodule objects."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/sub/__init__.py")
        _make_package(tmp_path, f"{pkg_name}/__init__.py", f"from {pkg_name} import sub\n")
        mod = __import__(pkg_name)
        assert "sub" not in originating_public_names(mod)

    def test_excludes_typing_type_checking(self, tmp_pkg: tuple[Path, str]) -> None:
        """originating_public_names excludes typing.TYPE_CHECKING."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(
            tmp_path,
            f"{pkg_name}/mod.py",
            "from typing import TYPE_CHECKING\n",
        )
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        assert "TYPE_CHECKING" not in originating_public_names(mod)


# --------------------------------------------------------------------------------------
# accessible_public_names
# --------------------------------------------------------------------------------------


class TestAccessiblePublicNames:
    """Tests for accessible_public_names."""

    def test_returns_public_attribute(self) -> None:
        """accessible_public_names returns attributes not starting with '_'."""
        mod = types.ModuleType("fake_mod")
        mod.VALUE = 42  # type: ignore[attr-defined]
        assert "VALUE" in accessible_public_names(mod)

    def test_excludes_private_attribute(self) -> None:
        """accessible_public_names excludes attributes starting with '_'."""
        mod = types.ModuleType("fake_mod")
        mod.__dict__["_hidden"] = 1
        assert "_hidden" not in accessible_public_names(mod)

    def test_excludes_submodule(self) -> None:
        """accessible_public_names excludes submodule objects."""
        parent = types.ModuleType("parent")
        child = types.ModuleType("parent.child")
        parent.child = child  # type: ignore[attr-defined]
        assert "child" not in accessible_public_names(parent)


# --------------------------------------------------------------------------------------
# find_names_missing_from_all
# --------------------------------------------------------------------------------------


class TestFindNamesMissingFromAll:
    """Tests for find_names_missing_from_all."""

    def test_regular_module_missing_name(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_names_missing_from_all reports originating names absent from __all__."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/mod.py", "class Defined:\n    pass\n")
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        assert "Defined" in find_names_missing_from_all(mod)

    def test_regular_module_declared_name_excluded(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_names_missing_from_all excludes names already listed in __all__."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(
            tmp_path,
            f"{pkg_name}/mod.py",
            '__all__ = ["Declared"]\nclass Declared:\n    pass\n',
        )
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        assert "Declared" not in find_names_missing_from_all(mod)

    def test_package_uses_accessible_names(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_names_missing_from_all uses accessible names (not just originating) for packages."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/__init__.py", "VALUE = 1\n")
        mod = __import__(pkg_name)
        assert "VALUE" in find_names_missing_from_all(mod)

    def test_package_declared_name_excluded(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_names_missing_from_all excludes package names already in __all__."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(
            tmp_path,
            f"{pkg_name}/__init__.py",
            '__all__ = ["VALUE"]\nVALUE = 1\n',
        )
        mod = __import__(pkg_name)
        assert "VALUE" not in find_names_missing_from_all(mod)

    def test_returns_sorted(self, tmp_pkg: tuple[Path, str]) -> None:
        """find_names_missing_from_all returns names in sorted order."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/mod.py", "Zebra = 1\nAlpha = 2\nMid = 3\n")
        mod = __import__(f"{pkg_name}.mod", fromlist=["mod"])
        missing = find_names_missing_from_all(mod)
        assert missing == sorted(missing)


# --------------------------------------------------------------------------------------
# collect_declared_obj_ids
# --------------------------------------------------------------------------------------


class TestCollectDeclaredObjIds:
    """Tests for collect_declared_obj_ids."""

    def test_returns_id_of_declared_symbol(self, tmp_pkg: tuple[Path, str]) -> None:
        """collect_declared_obj_ids returns the object ID of each symbol in __all__."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(
            tmp_path,
            f"{pkg_name}/__init__.py",
            '__all__ = ["VALUE"]\nVALUE = object()\n',
        )
        mod = importlib.import_module(pkg_name)
        ids = collect_declared_obj_ids([pkg_name])
        assert id(mod.VALUE) in ids

    def test_empty_for_no_all(self, tmp_pkg: tuple[Path, str]) -> None:
        """collect_declared_obj_ids returns an empty set when __all__ is absent."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/__init__.py", "VALUE = 1\n")
        assert len(collect_declared_obj_ids([pkg_name])) == 0

    def test_collects_across_multiple_packages(self, tmp_pkg: tuple[Path, str]) -> None:
        """collect_declared_obj_ids collects IDs from all provided packages."""
        tmp_path, pkg_name = tmp_pkg
        _make_package(tmp_path, f"{pkg_name}/__init__.py", '__all__ = ["A"]\nA = object()\n')
        _make_package(
            tmp_path,
            f"{pkg_name}/sub/__init__.py",
            '__all__ = ["B"]\nB = object()\n',
        )
        root = importlib.import_module(pkg_name)
        sub = importlib.import_module(f"{pkg_name}.sub")
        ids = collect_declared_obj_ids([pkg_name, f"{pkg_name}.sub"])
        assert id(root.A) in ids
        assert id(sub.B) in ids
