# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared test helpers for external/tests/devtools/.

These tests target standalone pre-commit scripts that aren't on the regular
import path (e.g. ``scripts/pre_commit/add_license_header.py``), so they load
each script via ``importlib.util.spec_from_file_location`` rather than a normal
import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_script(path: str | Path) -> ModuleType:
    """Import the script at ``path`` as a Python module.

    Registers the loaded module in ``sys.modules`` so ``@dataclass`` decorators
    paired with ``from __future__ import annotations`` can resolve their string
    type hints — without registration they fail with a confusing
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.

    Args:
        path (str | Path): Filesystem path to the script.

    Returns:
        ModuleType: The imported module.
    """
    script = Path(path)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
