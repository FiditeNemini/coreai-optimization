# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from pathlib import Path

import pytest
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

# Test-file-relative path so it works in both contexts: the internal layout
# (`external/tests/models/resnet.py` → `external/tests/test_utils/...`) and
# the OSS export layout (`tests/models/resnet.py` → `tests/test_utils/...`).
_TESTS_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def resnet50_model():
    """Fixture to provide a pre-trained ResNet50 model."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model


@pytest.fixture
def resnet18_model():
    """Fixture to provide a pre-trained ResNet18 model."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    return model


@pytest.fixture
def resnet_example_input():
    image_path = _TESTS_DIR / "test_utils" / "test_images" / "dog.jpg"
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
