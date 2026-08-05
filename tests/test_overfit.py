# Run this thang before any long run to check

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data.cifar10 import TRAINING_MEAN, TRAINING_STANDARD_DEVIATION
from models.resnet import CifarResNet
from train.utils import get_device, set_seed


IMAGE_COUNT = 100
MAX_ITERATIONS = 300
TARGET_ERROR = 0.02


def build_tiny_loader(data_directory="datasets", batch_size=50):
    dataset = datasets.CIFAR10(
        root=data_directory, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TRAINING_MEAN, TRAINING_STANDARD_DEVIATION),
        ]),
    )
    return DataLoader(
        Subset(dataset, list(range(IMAGE_COUNT))),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )


def overfit(residual=True):
    set_seed(0)
    device = get_device()

    model = CifarResNet(n=3, residual=residual).to(device)
    loader = build_tiny_loader()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    model.train()
    iteration = 0
    while iteration < MAX_ITERATIONS:
        for images, labels in loader:
            if iteration >= MAX_ITERATIONS:
                break

            images = images.to(device)
            labels = labels.to(device)

            loss = loss_function(model(images), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1

    model.eval()
    wrong = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1)
            wrong += (predictions != labels.to(device)).sum().item()
            total += labels.size(0)

    return wrong / total


def test_residual_network_memorises_one_hundred_images():
    error = overfit(residual=True)
    assert error <= TARGET_ERROR, (
        f"training error {error:.1%} after {MAX_ITERATIONS} iterations; "
        f"a working loop reaches under {TARGET_ERROR:.0%} on 100 images"
    )


def test_plain_network_memorises_one_hundred_images():
    error = overfit(residual=False)
    assert error <= TARGET_ERROR, f"training error {error:.1%}"


TESTS = [
    test_residual_network_memorises_one_hundred_images,
    test_plain_network_memorises_one_hundred_images,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except NotImplementedError:
            print(f"pending: {test.__name__}")
        except AssertionError as failure:
            print(f"failed: {test.__name__}: {failure}")
            failures += 1
        else:
            print(f"ok: {test.__name__}")

    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
