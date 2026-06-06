# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import torch.nn as nn

import tests.utils as utils
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
)

image_size = 28
batch_size = 128
num_classes = 10
num_epochs = 1


@pytest.mark.seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "spec",
    [
        PalettizationSpec(n_bits=2),
        PalettizationSpec(n_bits=4, cluster_dim=2),
        PalettizationSpec(
            n_bits=4,
            cluster_dim=2,
            granularity=PerGroupedChannelGranularity(axis=0, group_size=2),
        ),
        PalettizationSpec(
            n_bits=4,
            cluster_dim=2,
            granularity=PerGroupedChannelGranularity(axis=1, group_size=2),
        ),
    ],
)
def test_weight_only_ptq_mnist(mnist_pretrained_model, mnist_dataset, spec):
    """
    Train a simple convnet on the MNIST dataset for different deployment targets
    and verify its accuracy.

    Takes ~30s to run on an M1 Max Macbook Pro
    """
    # Setup pre-trained MNIST model
    model = mnist_pretrained_model

    # Setup test data loader for evaluation
    train_loader, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    # Verify baseline accuracy
    accuracy = utils.eval_model(model, test_loader)
    assert accuracy > 97.0, "expect pre-trained mnist model accuracy to be at least 97%"

    # Setup the quantizer
    config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": spec},
            enable_fast_kmeans_mode=False,
        ),
    )
    palettizer = KMeansPalettizer(model, config)

    prepared_model = palettizer.prepare(
        example_inputs=(torch.ones(1, 1, 28, 28, dtype=torch.float),),
        num_workers=1,
    )

    post_vanilla_kmeans_accuracy = utils.eval_model(prepared_model, test_loader)

    # Check that if there is any drop in accuracy, it is within 1%
    accuracy_drop = accuracy - post_vanilla_kmeans_accuracy
    assert accuracy_drop < 2, (
        f"Accuracy drop too high after vanilla kmeans: before={accuracy:.4f},"
        f"after={post_vanilla_kmeans_accuracy:.4f}"
    )

    with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy) as skm:
        for data, target in train_loader:
            output = prepared_model(data)
            skm.step(output, target)
            break

    post_skm_accuracy = utils.eval_model(prepared_model, test_loader)

    # Check that if there is any drop in accuracy, it is within 2% of the original
    accuracy_drop = accuracy - post_skm_accuracy
    assert accuracy_drop < 2, (
        f"Accuracy drop too high after SKM: before={accuracy:.4f},after={post_skm_accuracy:.4f}"
    )
