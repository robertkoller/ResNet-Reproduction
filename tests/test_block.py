import pathlib
import sys

# get our imports right
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from models.blocks import BasicBlock
from train.utils import set_seed


# Parameter counts we got from math
EXPECTED_PARAMETER_COUNTS = {
    (16, 16, 1): 4672,
    (16, 32, 2): 13952,
    (32, 32, 1): 18560,
    (32, 64, 2): 55552,
    (64, 64, 1): 73984,
}

STEM_PARAMETERS = 464
CLASSIFIER_PARAMETERS = 650
RESNET20_PARAMETERS = 269722


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def test_output_shape_is_unchanged_at_stride_one():
    block = BasicBlock(16, 16)
    output = block(torch.randn(2, 16, 32, 32))
    assert tuple(output.shape) == (2, 16, 32, 32), tuple(output.shape)


def test_parameter_count_of_a_16_to_16_block():
    count = count_parameters(BasicBlock(16, 16))
    assert count == 4672, f"got {count}, expected 4672 (4704 means bias=True on the convolutions)"


def test_residual_and_plain_have_identical_parameter_counts():
    residual = count_parameters(BasicBlock(16, 16, residual=True))
    plain = count_parameters(BasicBlock(16, 16, residual=False))
    assert residual == plain, f"residual {residual}, plain {plain}"


def test_every_block_type_matches_the_expected_count():
    for (in_channels, out_channels, stride), expected in EXPECTED_PARAMETER_COUNTS.items():
        block = BasicBlock(in_channels, out_channels, stride=stride)
        count = count_parameters(block)
        assert count == expected, f"{in_channels}->{out_channels} stride {stride}: got {count}, expected {expected}"


def test_blocks_sum_to_the_published_resnet20_total():
    # 0.27 miliii in the papers table, which is the architectures correctness proof
    stage1 = 3 * count_parameters(BasicBlock(16, 16))
    stage2 = count_parameters(BasicBlock(16, 32, stride=2)) + 2 * count_parameters(BasicBlock(32, 32))
    stage3 = count_parameters(BasicBlock(32, 64, stride=2)) + 2 * count_parameters(BasicBlock(64, 64))

    total = STEM_PARAMETERS + stage1 + stage2 + stage3 + CLASSIFIER_PARAMETERS
    assert total == RESNET20_PARAMETERS, f"got {total}, expected {RESNET20_PARAMETERS}"


def test_convolutions_carry_no_bias():
    block = BasicBlock(16, 16)
    assert block.convolution1.bias is None
    assert block.convolution2.bias is None


def test_normalization_layers_are_separate_instances():
    block = BasicBlock(16, 16)
    assert block.normalization1 is not block.normalization2


def test_output_is_finite():
    block = BasicBlock(16, 16)
    output = block(torch.randn(2, 16, 32, 32))
    assert torch.isfinite(output).all(), "output contains NaN or inf"


def test_plain_mode_changes_the_output():
    sample = torch.randn(2, 16, 32, 32)

    set_seed(0)
    residual_block = BasicBlock(16, 16, residual=True).eval()
    set_seed(0)
    plain_block = BasicBlock(16, 16, residual=False).eval()

    with torch.no_grad():
        assert not torch.allclose(residual_block(sample), plain_block(sample))


def test_gradients_reach_every_parameter():
    for residual in (True, False):
        block = BasicBlock(16, 16, residual=residual)
        block(torch.randn(2, 16, 32, 32)).sum().backward()

        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, f"residual={residual}: no gradient for {name}"
            assert torch.isfinite(parameter.grad).all(), f"residual={residual}: non-finite gradient for {name}"


def test_downsampling_without_a_shortcut_raises():
    block = BasicBlock(16, 32, stride=2)
    try:
        block(torch.randn(2, 16, 32, 32))
    except RuntimeError:
        return
    raise AssertionError("expected a RuntimeError from the shape mismatch")


def test_plain_mode_downsamples_without_a_shortcut():
    block = BasicBlock(16, 32, stride=2, residual=False)
    output = block(torch.randn(2, 16, 32, 32))
    assert tuple(output.shape) == (2, 32, 16, 16), tuple(output.shape)


TESTS = [
    test_output_shape_is_unchanged_at_stride_one,
    test_parameter_count_of_a_16_to_16_block,
    test_residual_and_plain_have_identical_parameter_counts,
    test_every_block_type_matches_the_expected_count,
    test_blocks_sum_to_the_published_resnet20_total,
    test_convolutions_carry_no_bias,
    test_normalization_layers_are_separate_instances,
    test_output_is_finite,
    test_plain_mode_changes_the_output,
    test_gradients_reach_every_parameter,
    test_downsampling_without_a_shortcut_raises,
    test_plain_mode_downsamples_without_a_shortcut,
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
