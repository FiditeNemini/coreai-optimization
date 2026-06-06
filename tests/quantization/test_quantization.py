# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Quantization tests parametrized across execution modes (graph and eager).

Each test class covers a distinct scenario or feature area of the quantizer.
Tests are parametrized via the ``execution_mode`` fixture so every scenario
runs for both graph and eager mode in a single test definition.  Where a mode
is known to be broken, ``pytest.xfail`` is used inline with an explanation.
"""

import pytest
import torch
import torch.nn as nn

from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization._graph.quantizer import GraphQuantizer
from coreai_opt.quantization.spec import (
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_EXECUTION_MODES = ["graph", "eager"]


def _make_w8a8_config(execution_mode: str) -> QuantizerConfig:
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        )
    ).set_execution_mode(execution_mode)


@pytest.fixture(params=_EXECUTION_MODES, ids=_EXECUTION_MODES)
def execution_mode(request) -> str:
    return request.param


@pytest.fixture
def example_input() -> torch.Tensor:
    return torch.randn(2, 8)


def _count_fake_quant_modules(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, FakeQuantizeImplBase))


def _make_module_name_config(
    execution_mode: str,
    module_name_configs: dict,
) -> QuantizerConfig:
    """W8A8 config with the given module_name_configs applied."""
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        ),
        module_name_configs=module_name_configs,
    ).set_execution_mode(execution_mode)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAliasedSubmoduleQuantization:
    """
    The quantizer must correctly handle models where a submodule is reachable
    under more than one attribute name — the pattern used by HuggingFace wrappers
    that hoist backbone children to the top level (e.g. ClipModule).

    Eager mode: expected to pass — hook-based matching is by object identity.
    Graph mode: currently expected to fail — alias name "encoder" ends up in
                module_configs (via named_children() recursion in
                _set_config_for_module) but is absent from
                module_name_to_state_names_map (built with named_modules()
                which deduplicates), causing an AssertionError inside
                _match_and_annotate_state_node.

    For the model used here:
      canonical name: "_model.encoder"
      alias name:     "encoder"
    """

    def _make_model(self) -> nn.Module:
        class _EncoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 8)
                self.fc2 = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(torch.relu(self.fc1(x)))

        class _BackboneModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _EncoderModel()
                self.proj = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(self.encoder(x))

        class AliasedSubmoduleModel(nn.Module):
            """
            Wrapper that stores a backbone as self._model and also hoists one of its
            children (encoder) as a top-level alias self.encoder — the same pattern
            used by HuggingFace model wrappers such as ClipModule.

            self._model   — the full backbone (registered first)
            self.encoder  — alias: same object as self._model.encoder
            """

            def __init__(self):
                super().__init__()
                self._model = _BackboneModel()
                self.encoder = self._model.encoder  # alias hoisted to top level
                self.head = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.encoder(x)  # uses the alias path
                x = self._model.proj(x)
                return self.head(x)

        return AliasedSubmoduleModel().eval()

    def _config_excluding_submodule(self, execution_mode: str, module_name: str) -> QuantizerConfig:
        """Global W8A8 config with the named submodule excluded (config=None)."""
        return QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec={"*": default_activation_quantization_spec()},
            ),
            module_name_configs={module_name: None},
        ).set_execution_mode(execution_mode)

    def test_fake_quant_inserted_in_aliased_submodule(self, execution_mode, example_input):
        """Fake-quant modules must be inserted for ops inside the aliased submodule."""
        prepared = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        assert _count_fake_quant_modules(prepared) > 0

    def test_canonical_and_alias_exclusion_same_result(self, execution_mode, example_input):
        """
        Excluding by canonical name and by alias name must produce the same
        fake-quant count — they refer to the same submodule.
        """
        full = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        excl_canonical = Quantizer(
            self._make_model(), self._config_excluding_submodule(execution_mode, "_model.encoder")
        ).prepare((example_input,))
        excl_alias = Quantizer(
            self._make_model(), self._config_excluding_submodule(execution_mode, "encoder")
        ).prepare((example_input,))
        assert _count_fake_quant_modules(excl_canonical) < _count_fake_quant_modules(full)
        assert _count_fake_quant_modules(excl_canonical) == _count_fake_quant_modules(excl_alias)

    def test_later_module_name_config_wins_over_earlier(self, execution_mode, example_input):
        """
        When module_name_configs contains multiple entries that resolve to the same
        module (via canonical name or alias), the later entry in the dict wins and
        is applied to all invocations of that module.

        We verify this by comparing against single-entry configs that represent
        what the winning entry alone would produce:
          - {"_model.encoder": None, "encoder": W8A8} → later W8A8 wins
            → same count as {"encoder": W8A8} alone
          - {"encoder": W8A8, "_model.encoder": None} → later None wins
            → same count as {"_model.encoder": None} alone
        """
        global_mod_config = ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        )

        # Later entry is W8A8 → equivalent to only having "encoder": W8A8
        later_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(
                execution_mode, {"_model.encoder": None, "encoder": global_mod_config}
            ),
        ).prepare((example_input,))
        only_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"encoder": global_mod_config}),
        ).prepare((example_input,))

        # Later entry is None → equivalent to only having "_model.encoder": None
        later_excl = Quantizer(
            self._make_model(),
            _make_module_name_config(
                execution_mode, {"encoder": global_mod_config, "_model.encoder": None}
            ),
        ).prepare((example_input,))
        only_excl = Quantizer(
            self._make_model(), _make_module_name_config(execution_mode, {"_model.encoder": None})
        ).prepare((example_input,))

        assert _count_fake_quant_modules(later_incl) == _count_fake_quant_modules(only_incl)
        assert _count_fake_quant_modules(later_excl) == _count_fake_quant_modules(only_excl)
        assert _count_fake_quant_modules(later_excl) < _count_fake_quant_modules(later_incl)


class TestReusedModuleQuantization:
    """
    The quantizer must correctly handle models where the same module object is
    used (called) multiple times — the pattern seen in EDSR where a single
    nn.ReLU is shared across 16 residual blocks.

    Here we use a shared nn.Linear so that the reused module is a quantizable op.

    ReusedLinearModel has:
      self.fc       — canonical name, same object as self.fc_alias
      self.fc_alias — alias, same object as self.fc

    forward() calls both self.fc(x) and self.fc_alias(x), producing two separate
    sets of nodes in the exported graph (one per call site).

    Key behaviors to verify:
    - Both call sites get fake-quant nodes (one per invocation, not shared).
    - module_name_config applies to ALL invocations regardless of which name
      (canonical or alias) is used — configuring by module object identity.
    - op_name_config (graph mode) can target individual call-site nodes
      independently, enabling per-invocation configuration.
    """

    def _make_model(self) -> nn.Module:
        class ReusedLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)
                self.fc_alias = self.fc  # same object, different name
                self.head = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc(x)  # first invocation
                x = self.fc_alias(x)  # second invocation (same module)
                return self.head(x)

        return ReusedLinearModel().eval()

    def _config_excluding_module(self, execution_mode: str, module_name: str) -> QuantizerConfig:
        return QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec={"*": default_activation_quantization_spec()},
            ),
            module_name_configs={module_name: None},
        ).set_execution_mode(execution_mode)

    def test_fake_quant_inserted_for_each_invocation(self, execution_mode, example_input):
        """
        Each call site of the shared module must get its own fake-quant nodes.
        Excluding the shared module by name must reduce the count compared to
        the fully-quantized baseline.
        """
        full = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        excl = Quantizer(
            self._make_model(), self._config_excluding_module(execution_mode, "fc")
        ).prepare((example_input,))
        assert _count_fake_quant_modules(excl) < _count_fake_quant_modules(full)

    def test_module_name_config_any_alias_excludes_all_invocations(
        self, execution_mode, example_input
    ):
        """
        Excluding by canonical name ('fc') or alias name ('fc_alias') must
        both exclude ALL invocations and produce the same fake-quant count —
        module_name_config applies by module object identity.
        """
        full = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        excl_canonical = Quantizer(
            self._make_model(), self._config_excluding_module(execution_mode, "fc")
        ).prepare((example_input,))
        excl_alias = Quantizer(
            self._make_model(), self._config_excluding_module(execution_mode, "fc_alias")
        ).prepare((example_input,))
        assert _count_fake_quant_modules(full) > _count_fake_quant_modules(excl_canonical)
        assert _count_fake_quant_modules(excl_canonical) == _count_fake_quant_modules(excl_alias)

    def test_op_name_config_allows_per_invocation_config(self, example_input):
        """
        op_name_config targets individual graph nodes by name, enabling
        per-invocation configuration of a shared module (graph mode only).

        We prepare once to discover node names, then build a config that
        excludes just one of the two fc invocations and verify the fake-quant
        count falls strictly between the full and fully-excluded counts.
        """

        # Discover compressible op names from the prepared graph
        discovery_prepared = Quantizer(self._make_model(), _make_w8a8_config("graph")).prepare(
            (example_input,)
        )
        compressible = GraphQuantizer.get_compressible_op_names(discovery_prepared)
        assert len(compressible) >= 2, f"Expected >=2 compressible ops, got: {compressible}"

        # Exclude just the first op via op_name_config
        first_op = sorted(compressible)[0]
        op_excl_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec={"*": default_activation_quantization_spec()},
                op_name_config={first_op: None},
            )
        ).set_execution_mode("graph")

        full = Quantizer(self._make_model(), _make_w8a8_config("graph")).prepare((example_input,))
        excl_one = Quantizer(self._make_model(), op_excl_config).prepare((example_input,))
        excl_all = Quantizer(
            self._make_model(), self._config_excluding_module("graph", "fc")
        ).prepare((example_input,))

        # Excluding one invocation: count strictly between full and fully-excluded
        assert _count_fake_quant_modules(excl_all) < _count_fake_quant_modules(excl_one)
        assert _count_fake_quant_modules(excl_one) < _count_fake_quant_modules(full)

    def test_later_module_name_config_wins_all_invocations(self, execution_mode, example_input):
        """
        When module_name_configs contains multiple entries resolving to the same
        shared module, the later entry wins and applies to ALL call sites of that
        module (both invocations of self.fc and self.fc_alias).

        Verified by comparing against single-entry equivalent configs:
          - {"fc": None, "fc_alias": W8A8} → later W8A8 wins → same as {"fc_alias": W8A8}
          - {"fc_alias": W8A8, "fc": None} → later None wins → same as {"fc": None}
        """
        global_mod_config = ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        )

        later_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"fc": None, "fc_alias": global_mod_config}),
        ).prepare((example_input,))
        only_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"fc_alias": global_mod_config}),
        ).prepare((example_input,))

        later_excl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"fc_alias": global_mod_config, "fc": None}),
        ).prepare((example_input,))
        only_excl = Quantizer(
            self._make_model(), _make_module_name_config(execution_mode, {"fc": None})
        ).prepare((example_input,))

        assert _count_fake_quant_modules(later_incl) == _count_fake_quant_modules(only_incl)
        assert _count_fake_quant_modules(later_excl) == _count_fake_quant_modules(only_excl)
        assert _count_fake_quant_modules(later_excl) < _count_fake_quant_modules(later_incl)
