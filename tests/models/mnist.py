# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from filelock import FileLock
from torchvision import datasets, transforms

from tests.utils import test_artifact_path, test_data_path


@pytest.fixture(scope="function")
def custom_test_mnist_model(num_classes: int = 10):
    """
    This is a custom toy model designed to include layers with parameters like conv,
    conv transpose, and linear. It’s intended for testing on the
    MNIST 28x28 image dataset.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 32, (5, 5), padding=2)),
                ("bn1", nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
                ("relu1", nn.ReLU6()),
                ("pool1", nn.MaxPool2d(2, stride=2, padding=0)),
                ("conv2", nn.Conv2d(32, 64, (5, 5), padding=2)),
                ("relu2", nn.ReLU6()),
                ("pool2", nn.MaxPool2d(2, stride=2, padding=0)),
                ("conv_transpose1", nn.ConvTranspose2d(64, 128, 3, padding=1)),
                ("bn3", nn.BatchNorm2d(128, eps=0.001, momentum=0.01)),
                ("relu3", nn.ReLU6()),
                ("pool3", nn.MaxPool2d(1, stride=1, padding=0)),
                ("conv_transpose2", nn.ConvTranspose2d(128, 64, 3, padding=1)),
                ("relu4", nn.ReLU6()),
                ("flatten", nn.Flatten()),
                ("dense1", nn.Linear(3136, 1024)),
                ("relu5", nn.ReLU6()),
                ("dropout", nn.Dropout(p=0.4)),
                ("dense2", nn.Linear(1024, num_classes)),
                ("softmax", nn.LogSoftmax(dim=-1)),
            ]
        )
    )


@pytest.fixture(scope="session")
def mnist_data(temp_dir):
    """Fixture to provide MNIST test data."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load only a small subset of MNIST test data for testing purposes
    test_data_path = Path(temp_dir) / "mnist_data"
    test_dataset = datasets.MNIST(test_data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Return a single batch
    for data, target in test_loader:
        return data, target


@pytest.fixture(scope="session")
def mnist_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_path = Path(test_data_path()) / "mnist"
    data_path.mkdir(parents=True, exist_ok=True)
    lock_file = data_path / "data.lock"
    with FileLock(str(lock_file)):
        train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    return train, test


@pytest.fixture()
def mnist_example_input():
    """Load the committed MNIST example-input artifact from `_test_artifacts/`."""
    return torch.load(test_artifact_path("mnist/mnist_example_input_11122025.pt"))


@pytest.fixture()
def mnist_example_output(num_classes: int = 10):
    return torch.rand(1, num_classes)
