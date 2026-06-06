# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for config related items"""
from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

# Constant representing all tensors for an input/output/state spec
ALL_TENSORS = "*"


def is_yaml_file(file_path: Path) -> bool:
    """
    Returns True if file_path points to a file ending in .yaml or .yml suffix, False
    otherwise.
    """
    return file_path.is_file() and file_path.suffix.lower() in ['.yaml', '.yml']


class ConfigLevel(Enum):
    """
    Enum to specify the config type.

    Enum entries should be defined in order of highest priority to lowest priority.

    - MODULE_NAME: Applied to specific module names (e.g., "layer1.conv")
    - MODULE_TYPE: Applied to specific module types (e.g., all Conv2d)
    - GLOBAL: Applied to all modules
    """
    MODULE_NAME = auto()
    MODULE_TYPE = auto()
    GLOBAL = auto()

    @classmethod
    def priority_order(cls) -> list[ConfigLevel]:
        """Return config levels in priority order (highest to lowest)."""
        return list(cls)
