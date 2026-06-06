# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""End-to-end MNIST tests for MagnitudePruner."""

import copy

import pytest
import torch

import tests.utils as utils
from coreai_opt.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PruningSpec,
)
from coreai_opt.pruning.config import PolynomialDecaySchedule

batch_size = 128


@pytest.mark.slow
@pytest.mark.seed
def test_magnitude_pruner_mnist_e2e(mnist_pretrained_model, mnist_dataset) -> None:
    """End-to-end MNIST: scheduled pruning with fine-tuning vs post-training static pruning.

    Establishes a baseline MNIST model. Then:

    1. Post-training static pruning at 70% — no fine-tuning, no schedule.
       Measures the accuracy degradation when 70% of weights are zeroed out instantly.
    2. On a fresh copy, applies a polynomial sparsity schedule that ramps to
       70% over several training epochs with fine-tuning between sparsity bumps.

    The scheduled-pruning model should recover noticeably more accuracy than
    the static-pruning baseline.
    """
    target = 0.7
    num_epochs = 5
    example_inputs = (torch.ones(1, 1, 28, 28, dtype=torch.float),)
    train_loader, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    baseline_acc = utils.eval_model(mnist_pretrained_model, test_loader)
    assert baseline_acc > 97.0, "pre-trained MNIST baseline should be > 97%"

    # 1. Static pruning: apply target sparsity in one shot, no fine-tuning.
    static_model = copy.deepcopy(mnist_pretrained_model)
    static_pruner = MagnitudePruner(
        static_model,
        MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=target)},
            ),
        ),
    )
    static_pruner.prepare(example_inputs)
    static_acc = utils.eval_model(static_model, test_loader)
    assert static_acc < baseline_acc, "static pruning at 70% should degrade accuracy"

    # 2. Scheduled pruning: ramp to target over num_epochs, fine-tuning between bumps.
    scheduled_model = copy.deepcopy(mnist_pretrained_model)
    scheduled_pruner = MagnitudePruner(
        scheduled_model,
        MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=target)},
                sparsity_schedule=PolynomialDecaySchedule(
                    begin_step=0, total_iters=num_epochs, power=3.0
                ),
            ),
        ),
    )
    scheduled_pruner.prepare(example_inputs)

    optimizer = torch.optim.SGD(scheduled_model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        scheduled_model.train()
        for batch_idx, (data, target_y) in enumerate(train_loader):
            utils.train_step(
                scheduled_model, optimizer, train_loader, data, target_y, batch_idx, epoch
            )
        scheduled_pruner.step()

    # Every parametrized module's weight should have ~target_sparsity zeros.
    for _, module in scheduled_model.named_modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            zero_fraction = module.weight.eq(0.0).sum() / module.weight.numel()
            # Tolerance covers floor rounding when numel * target isn't integer.
            assert zero_fraction.item() == pytest.approx(target, abs=1e-3)

    scheduled_acc = utils.eval_model(scheduled_model, test_loader)
    assert scheduled_acc > static_acc, (
        f"scheduled pruning ({scheduled_acc:.2f}%) should recover more accuracy than "
        f"static pruning ({static_acc:.2f}%); baseline was {baseline_acc:.2f}%"
    )
