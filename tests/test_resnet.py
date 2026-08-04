import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from models.blocks import BottleneckBlock
from models.counters import count_flops, count_parameters
from models.resnet import CifarResNet, ImageNetResNet, count_layers


CIFAR_PARAMETERS = {
    3: 269722,
    5: 464154,
    7: 658586,
    9: 853018,
    18: 1727962,
}

IMAGENET_PARAMETERS = {
    18: 11689512,
    34: 21797672,
    50: 25557032,
    101: 44549160,
    152: 60192808,
}


def test_cifar_parameter_counts_match_the_paper():
    for n, expected in CIFAR_PARAMETERS.items():
        count = count_parameters(CifarResNet(n=n))
        assert count == expected, f"n={n}: got {count:,}, expected {expected:,}"


def test_cifar_layer_counts_are_six_n_plus_two():
    for n in CIFAR_PARAMETERS:
        model = CifarResNet(n=n)
        layers = count_layers(model)
        assert layers == 6 * n + 2, f"n={n}: counted {layers} layers, expected {6 * n + 2}"


def test_cifar_output_shape():
    model = CifarResNet(n=3)
    output = model(torch.randn(4, 3, 32, 32))
    assert tuple(output.shape) == (4, 10), tuple(output.shape)
    assert torch.isfinite(output).all()


def test_residual_and_plain_match_under_option_a():
    for n in (3, 9):
        residual = count_parameters(CifarResNet(n=n, residual=True, shortcut="A"))
        plain = count_parameters(CifarResNet(n=n, residual=False, shortcut="A"))
        assert residual == plain, f"n={n}: residual {residual:,}, plain {plain:,}"


def test_shortcut_options_add_parameters_in_the_expected_order():
    counts = {
        option: count_parameters(CifarResNet(n=3, shortcut=option))
        for option in ("A", "B", "C")
    }
    assert counts["A"] < counts["B"] < counts["C"], counts


def test_intermediate_shapes():
    model = CifarResNet(n=3).eval()
    expected = [
        ("stage1", (2, 16, 32, 32)),
        ("stage2", (2, 32, 16, 16)),
        ("stage3", (2, 64, 8, 8)),
    ]

    with torch.no_grad():
        out = model.relu(model.stem_normalization(model.stem_convolution(torch.randn(2, 3, 32, 32))))
        assert tuple(out.shape) == (2, 16, 32, 32), tuple(out.shape)
        for name, shape in expected:
            out = getattr(model, name)(out)
            assert tuple(out.shape) == shape, f"{name}: {tuple(out.shape)}, expected {shape}"


def test_gradients_reach_every_parameter():
    for residual in (True, False):
        model = CifarResNet(n=3, residual=residual)
        model(torch.randn(2, 3, 32, 32)).sum().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"residual={residual}: no gradient for {name}"


def test_he_initialization_variance():
    model = CifarResNet(n=9)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        fan_out = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        expected = math.sqrt(2.0 / fan_out)
        actual = module.weight.std().item()
        assert abs(actual - expected) / expected < 0.15, (
            f"{name}: std {actual:.4f}, expected about {expected:.4f}"
        )


def test_batch_norms_start_as_the_identity():
    model = CifarResNet(n=3)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.all(module.weight == 1.0), name
            assert torch.all(module.bias == 0.0), name


def test_imagenet_parameter_counts():
    for depth, expected in IMAGENET_PARAMETERS.items():
        count = count_parameters(ImageNetResNet(depth))
        assert count == expected, f"resnet{depth}: got {count:,}, expected {expected:,}"


def test_imagenet_output_shape_and_bottleneck_expansion():
    model = ImageNetResNet(50, num_classes=10)
    output = model(torch.randn(2, 3, 224, 224))
    assert tuple(output.shape) == (2, 10), tuple(output.shape)
    assert BottleneckBlock.expansion == 4
    assert model.classifier.in_features == 512 * 4


def test_unknown_imagenet_depth_is_rejected():
    try:
        ImageNetResNet(42)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for an unsupported depth")


def test_flop_counts_are_in_the_published_range():
    for depth, published in ((18, 1.8e9), (34, 3.6e9)):
        flops = count_flops(ImageNetResNet(depth), (3, 224, 224))
        assert abs(flops - published) / published < 0.05, (
            f"resnet{depth}: {flops / 1e9:.2f}G, paper reports {published / 1e9:.1f}G"
        )


def test_cifar_flops_scale_with_depth():
    previous = 0
    for n in (3, 5, 7, 9, 18):
        flops = count_flops(CifarResNet(n=n), (3, 32, 32))
        assert flops > previous, f"n={n} did not increase FLOPs"
        previous = flops


TESTS = [
    test_cifar_parameter_counts_match_the_paper,
    test_cifar_layer_counts_are_six_n_plus_two,
    test_cifar_output_shape,
    test_residual_and_plain_match_under_option_a,
    test_shortcut_options_add_parameters_in_the_expected_order,
    test_intermediate_shapes,
    test_gradients_reach_every_parameter,
    test_he_initialization_variance,
    test_batch_norms_start_as_the_identity,
    test_imagenet_parameter_counts,
    test_imagenet_output_shape_and_bottleneck_expansion,
    test_unknown_imagenet_depth_is_rejected,
    test_flop_counts_are_in_the_published_range,
    test_cifar_flops_scale_with_depth,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as failure:
            print(f"failed: {test.__name__}: {failure}")
            failures += 1
        else:
            print(f"ok: {test.__name__}")

    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
