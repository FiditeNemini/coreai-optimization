# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ExecutionMode, including deprecated alias resolution."""

from __future__ import annotations

import warnings

import pytest

from coreai_opt.quantization.config import ExecutionMode
from coreai_opt.quantization.config.quantization_config import QuantizerConfig


class TestExecutionMode:
    """Construction and deprecation behavior for ExecutionMode."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("graph", ExecutionMode.GRAPH),
            ("eager", ExecutionMode.EAGER),
        ],
    )
    def test_construct_from_current_value_does_not_warn(
        self,
        value: str,
        expected: ExecutionMode,
    ) -> None:
        """Resolve current values without emitting a DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert ExecutionMode(value) is expected

    @pytest.mark.parametrize(
        ("old_name", "expected"),
        [
            ("PT2E", ExecutionMode.GRAPH),
        ],
    )
    def test_deprecated_alias_warns_and_resolves(
        self,
        old_name: str,
        expected: ExecutionMode,
    ) -> None:
        """Resolve deprecated names via attribute access and value lookup."""
        new_name = expected.name
        old_value = old_name.lower()

        with pytest.warns(
            DeprecationWarning,
            match=rf"ExecutionMode\.{old_name} is deprecated",
        ) as record:
            assert getattr(ExecutionMode, old_name) is expected
        assert f"ExecutionMode.{new_name}" in str(record[0].message)

        with pytest.warns(
            DeprecationWarning,
            match=rf"ExecutionMode\('{old_value}'\) is deprecated",
        ) as record:
            assert ExecutionMode(old_value) is expected
        assert f"ExecutionMode.{new_name}" in str(record[0].message)

    def test_deprecated_literal_attribute_access_resolves_for_clients(self) -> None:
        """Verify literal `ExecutionMode.PT2E` (what client code writes) still works."""
        with pytest.warns(
            DeprecationWarning,
            match=r"PT2E is deprecated, use ExecutionMode\.GRAPH instead",
        ):
            assert ExecutionMode.PT2E is ExecutionMode.GRAPH

    def test_deprecated_yaml_value_resolves_via_quantizer_config(self) -> None:
        """Verify old YAML value ``execution_mode: pt2e`` still deserializes to GRAPH.

        User-facing YAML configs are parsed by Pydantic through
        ``QuantizerConfig.from_dict``, which invokes ``ExecutionMode("pt2e")``. This
        path must continue to resolve (with a DeprecationWarning) so existing user
        configs keep working until the alias is removed.
        """
        with pytest.warns(
            DeprecationWarning,
            match=r"ExecutionMode\('pt2e'\) is deprecated",
        ):
            config = QuantizerConfig.from_dict({"quantization_config": {"execution_mode": "pt2e"}})
        assert config.execution_mode is ExecutionMode.GRAPH
