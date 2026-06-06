# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for inspecting and enforcing public API visibility.

Used by tests/test_api_visibility.py and scripts/make/print_api_list.py to discover public
packages and modules, identify symbols originating in a module vs. imported, and find symbols
missing from __all__.
"""

import importlib
import pkgutil
import types
import typing
from collections.abc import Sequence
from pathlib import Path

_SENTINEL = object()

# Stdlib modules whose names should be treated as imports rather than originating here, even
# when they have no `__module__` attribute (e.g. bare booleans like `typing.TYPE_CHECKING`).
_STDLIB_IDENTITY_SOURCES: tuple[types.ModuleType, ...] = (typing,)


def find_public_packages(root_package: str) -> list[str]:
    """Return all public package names under root_package.

    A package is public if no segment of its dotted import path (after the root package) starts
    with '_'.
    """
    return [root_package, *_walk_public_names(root_package, packages=True)]


def find_public_modules(root_package: str) -> list[str]:
    """Return all public non-package module names under root_package.

    A module is public if no segment of its dotted import path (after the root package) starts
    with '_'. Only non-package .py files are returned (i.e., files that are not `__init__.py`).
    """
    return _walk_public_names(root_package, packages=False)


def originating_public_names(mod: types.ModuleType) -> set[str]:
    """Return public names that originate in a module, not imported from elsewhere.

    A name is considered to originate in the module if:
    - It does not start with `_`
    - It is not a module object
    - It does not contain `[` (filters Pydantic generic specializations)
    - For classes and functions: `__module__` matches the module name
    - For other values (constants, TypeVars, etc.): the value is present in the module's
      `__dict__` and has no `__module__` pointing elsewhere. Bare primitives imported from
      stdlib (e.g. ``typing.TYPE_CHECKING``) are further filtered by identity, since they
      have no `__module__` at all.
    """
    mod_name = mod.__name__
    mod_dict = vars(mod)
    result: set[str] = set()
    for name in dir(mod):
        if name.startswith("_") or "[" in name:
            continue
        if name not in mod_dict:
            continue
        obj = mod_dict[name]
        if isinstance(obj, types.ModuleType):
            continue
        origin = getattr(obj, "__module__", None)
        if isinstance(obj, (type, types.FunctionType)):
            # Classes and functions carry __module__; only keep if it matches.
            if origin == mod_name:
                result.add(name)
        elif origin == mod_name:
            result.add(name)
        elif origin is None and not _is_stdlib_identity_reexport(name, obj):
            result.add(name)
    return result


def _is_stdlib_identity_reexport(name: str, obj: object) -> bool:
    """Return True if `obj` is the same object as an attribute of a known stdlib module.

    Identity (``is``), not equality (``==``): the identity check requires both a name match
    and pointer identity, so a module-defined ``flag = True`` does not accidentally match
    ``typing.TYPE_CHECKING`` just because both equal ``True``.
    """
    return any(getattr(source, name, _SENTINEL) is obj for source in _STDLIB_IDENTITY_SOURCES)


def accessible_public_names(mod: types.ModuleType) -> set[str]:
    """Return public, non-submodule names accessible on a module via `dir()`.

    A name is included if it does not start with '_' and its value is not a module object.
    """
    return {
        name
        for name in dir(mod)
        if not name.startswith("_") and not isinstance(getattr(mod, name), types.ModuleType)
    }


def find_names_missing_from_all(mod: types.ModuleType) -> list[str]:
    """Return sorted public names not listed in __all__.

    For packages (`__init__.py`): returns accessible non-submodule names missing from `__all__`
    (i.e. names reachable via `dir()`).

    For regular modules: returns names that originate in the module (not imported from elsewhere)
    that are missing from `__all__`.
    """
    all_names = set(getattr(mod, "__all__", []))
    mod_file = getattr(mod, "__file__", None) or ""
    if mod_file.endswith("__init__.py"):
        return sorted(accessible_public_names(mod) - all_names)
    return sorted(originating_public_names(mod) - all_names)


def collect_declared_obj_ids(package_names: Sequence[str]) -> set[int]:
    """Return object IDs of all symbols declared in __all__ across the given modules.

    Used to check whether a symbol defined in a module is already re-exported via some package's
    __all__.
    """
    return set(collect_declared_obj_id_map(package_names).keys())


def collect_declared_obj_id_map(package_names: Sequence[str]) -> dict[int, str]:
    """Return a map from object ID to the first declaring package name.

    For each symbol listed in any package's __all__, records the object ID mapped to the dotted
    name of the package that declares it. When the same object is declared in multiple packages,
    the first one encountered wins.
    """
    declared: dict[int, str] = {}
    for pkg_name in package_names:
        mod = importlib.import_module(pkg_name)
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            if obj is not None and id(obj) not in declared:
                declared[id(obj)] = pkg_name
    return declared


def _walk_public_names(root_package: str, *, packages: bool) -> list[str]:
    """Return sorted public package or module names under root_package.

    When `packages` is True, returns sub-packages; when False, returns non-package modules. In
    both cases, only names whose every path segment (after the root) is public (no leading '_')
    are included.
    """
    root_mod = importlib.import_module(root_package)
    if root_mod.__file__ is None:
        msg = f"{root_package} is not a package"
        raise RuntimeError(msg)
    root_path = Path(root_mod.__file__).parent

    prefix = f"{root_package}."
    result: list[str] = []
    for _importer, name, is_pkg in pkgutil.walk_packages(
        path=[str(root_path)],
        prefix=prefix,
        onerror=lambda _: None,
    ):
        if is_pkg != packages:
            continue
        relative_parts = name.removeprefix(prefix).split(".")
        if any(part.startswith("_") for part in relative_parts):
            continue
        result.append(name)
    return sorted(result)
