"""The training loop.

The schedule is specified in iterations rather than epochs, because that is how
the paper specifies it: 64,000 steps with the learning rate divided by ten at
32,000 and again at 48,000. An epoch is 351 iterations at batch size 128, so
the loop runs the data loader repeatedly and counts steps, rather than counting
epochs and hoping the arithmetic lines up.
"""

import csv
import pathlib
import time

import torch
from torch import nn

from data.cifar10 import build_dataloaders
from models.counters import count_parameters
from models.resnet import CifarResNet
from train.utils import get_device


METRICS_HEADER = (
    "iteration",
    "learning_rate",
    "train_loss",
    "train_error",
    "validation_error",
    "elapsed_seconds",
)


def build_optimizer(model, configuration):
    return torch.optim.SGD(
        model.parameters(),
        lr=configuration.learning_rate,
        momentum=configuration.momentum,
        weight_decay=configuration.weight_decay,
    )


def current_learning_rate(configuration, iteration, warmup_finished):
    if configuration.warmup and not warmup_finished:
        return configuration.warmup_learning_rate

    learning_rate = configuration.learning_rate
    for milestone in configuration.learning_rate_milestones:
        if iteration >= milestone:
            learning_rate *= configuration.learning_rate_gamma

    return learning_rate


def set_learning_rate(optimizer, learning_rate):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    # Without eval() BatchNorm uses this batch's statistics instead of the
    # running ones, which quietly produces the wrong number.
    model.eval()

    wrong = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images).argmax(dim=1)
        wrong += (predictions != labels).sum().item()
        total += labels.size(0)

    model.train(was_training)
    return wrong / total


def open_metrics_file(path, append=False):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existed = path.exists()
    handle = open(path, "a" if append else "w", newline="")
    writer = csv.writer(handle)
    if not (append and existed):
        writer.writerow(METRICS_HEADER)
    return handle, writer


def save_checkpoint(path, model, optimizer, iteration, warmup_finished):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            # Not optional: SGD carries a momentum buffer per parameter, and
            # dropping it makes training stutter for several hundred steps.
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "warmup_finished": warmup_finished,
        },
        path,
    )
    return path


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"], checkpoint["warmup_finished"]


def train(configuration, resume=False):
    device = get_device()
    model = CifarResNet(
        n=configuration.n,
        residual=configuration.residual,
        shortcut=configuration.shortcut,
    ).to(device)

    training_loader, validation_loader, test_loader = build_dataloaders(configuration)
    optimizer = build_optimizer(model, configuration)
    loss_function = nn.CrossEntropyLoss()

    checkpoint_path = configuration.run_directory / "checkpoint.pt"
    iteration = 0
    warmup_finished = not configuration.warmup

    if resume:
        if checkpoint_path.exists():
            iteration, warmup_finished = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            print(f"resumed from iteration {iteration}")
        else:
            print(f"no checkpoint at {checkpoint_path}, starting from scratch")

    configuration.save()
    handle, writer = open_metrics_file(
        configuration.run_directory / "metrics.csv", append=iteration > 0
    )

    print(f"depth {configuration.depth}, {count_parameters(model):,} parameters, device {device}")

    started = time.perf_counter()
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_examples = 0
    running_steps = 0

    while iteration < configuration.max_iterations:
        for images, labels in training_loader:
            if iteration >= configuration.max_iterations:
                break

            set_learning_rate(
                optimizer,
                current_learning_rate(configuration, iteration, warmup_finished),
            )

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_correct = (outputs.argmax(dim=1) == labels).sum().item()
            running_loss += loss.item()
            running_correct += batch_correct
            running_examples += labels.size(0)
            running_steps += 1

            if not warmup_finished:
                batch_error = 1.0 - batch_correct / labels.size(0)
                if batch_error < configuration.warmup_error_threshold:
                    warmup_finished = True
                    print(
                        f"warmup finished at iteration {iteration} "
                        f"(batch error {batch_error:.1%})"
                    )

            iteration += 1

            if iteration % configuration.evaluate_every == 0:
                train_error = 1.0 - running_correct / running_examples
                train_loss = running_loss / running_steps
                validation_error = evaluate(model, validation_loader, device)
                elapsed = time.perf_counter() - started
                learning_rate = current_learning_rate(
                    configuration, iteration, warmup_finished
                )

                writer.writerow(
                    [
                        iteration,
                        learning_rate,
                        round(train_loss, 6),
                        round(train_error, 6),
                        round(validation_error, 6),
                        round(elapsed, 2),
                    ]
                )
                handle.flush()

                print(
                    f"  iter {iteration:>6}/{configuration.max_iterations}  "
                    f"lr {learning_rate:<7.4g} loss {train_loss:.4f}  "
                    f"train err {train_error:.2%}  val err {validation_error:.2%}  "
                    f"{elapsed:.0f}s"
                )

                running_loss = 0.0
                running_correct = 0
                running_examples = 0
                running_steps = 0

            if iteration % configuration.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_path, model, optimizer, iteration, warmup_finished
                )

    handle.close()
    save_checkpoint(checkpoint_path, model, optimizer, iteration, warmup_finished)

    test_error = evaluate(model, test_loader, device)
    validation_error = evaluate(model, validation_loader, device)
    print(
        f"final: test error {test_error:.2%}, validation error {validation_error:.2%}, "
        f"{time.perf_counter() - started:.0f}s"
    )
    return test_error
