"""CIFAR-10 loading, splitting, augmentation and normalization.

The dataset ships as 50,000 training and 10,000 test images. 5,000 of the
training images are held out for validation, so every decision made during
development is checked against data the test set never sees.
"""

import json
import pathlib
import random

import numpy
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TRAINING_MEAN = (0.4915, 0.4821, 0.4464)
TRAINING_STANDARD_DEVIATION = (0.2469, 0.2435, 0.2614)

VALIDATION_SIZE = 5000
IMAGE_SIZE = 32
PADDING = 4

SPLIT_SEED = 0


def make_splits(dataset_size=50000, validation_size=VALIDATION_SIZE, seed=SPLIT_SEED):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()

    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    return training_indices, validation_indices


def save_splits(training_indices, validation_indices, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {"train": training_indices, "validation": validation_indices},
            handle,
        )
    return path


def compute_normalization_statistics(data_directory):
    dataset = datasets.CIFAR10(
        root=data_directory, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    training_indices, _ = make_splits()
    training_images = Subset(dataset, training_indices)

    # Accumulate rather than stacking 45,000 images at once.
    channel_sum = torch.zeros(3)
    channel_square_sum = torch.zeros(3)
    pixel_count = 0

    for image, _ in training_images:
        channel_sum += image.sum(dim=(1, 2))
        channel_square_sum += (image ** 2).sum(dim=(1, 2))
        pixel_count += image.shape[1] * image.shape[2]

    mean = channel_sum / pixel_count
    variance = channel_square_sum / pixel_count - mean ** 2

    return tuple(mean.tolist()), tuple(variance.sqrt().tolist())


def build_transforms(training):
    steps = []

    if training:
        steps.append(transforms.RandomCrop(IMAGE_SIZE, padding=PADDING))
        steps.append(transforms.RandomHorizontalFlip())

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(TRAINING_MEAN, TRAINING_STANDARD_DEVIATION))

    return transforms.Compose(steps)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(configuration, num_workers=2):
    data_directory = configuration.data_directory
    training_indices, validation_indices = make_splits()

    augmented = datasets.CIFAR10(
        root=data_directory, train=True, download=True,
        transform=build_transforms(training=True),
    )
    unaugmented = datasets.CIFAR10(
        root=data_directory, train=True, download=True,
        transform=build_transforms(training=False),
    )

    training_dataset = Subset(augmented, training_indices)
    validation_dataset = Subset(unaugmented, validation_indices)
    test_dataset = datasets.CIFAR10(
        root=data_directory, train=False, download=True,
        transform=build_transforms(training=False),
    )

    shuffle_generator = torch.Generator().manual_seed(configuration.seed)

    # Worker startup costs about 7 seconds each on macOS, and workers are torn
    # down at the end of every epoch unless kept alive. Over 182 epochs that
    # would be tens of minutes of pure overhead.
    persistent = num_workers > 0

    training_loader = DataLoader(
        training_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=shuffle_generator,
        worker_init_fn=seed_worker,
        persistent_workers=persistent,
    )

    evaluation_batch_size = max(configuration.batch_size, 256)
    validation_loader = DataLoader(
        validation_dataset, batch_size=evaluation_batch_size,
        shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=evaluation_batch_size,
        shuffle=False, num_workers=num_workers,
    )

    return training_loader, validation_loader, test_loader
