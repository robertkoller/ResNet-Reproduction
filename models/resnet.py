import torch
from torch import nn

from models.blocks import BasicBlock, BottleneckBlock, make_shortcut


STAGE_CHANNELS = (16, 32, 64)
STEM_CHANNELS = 16

IMAGENET_STAGE_CHANNELS = (64, 128, 256, 512)
IMAGENET_STEM_CHANNELS = 64

IMAGENET_LAYOUTS = {
    18: (BasicBlock, (2, 2, 2, 2)),
    34: (BasicBlock, (3, 4, 6, 3)),
    50: (BottleneckBlock, (3, 4, 6, 3)),
    101: (BottleneckBlock, (3, 4, 23, 3)),
    152: (BottleneckBlock, (3, 8, 36, 3)),
}


def build_stage(in_channels, out_channels, block_count, stride, residual,
                shortcut_option, block_class=BasicBlock):
    blocks = []
    expanded_channels = out_channels * block_class.expansion

    for index in range(block_count):
        block_in_channels = in_channels if index == 0 else expanded_channels
        block_stride = stride if index == 0 else 1

        if residual:
            shortcut = make_shortcut(block_in_channels, expanded_channels, block_stride, shortcut_option)
        else:
            shortcut = None

        blocks.append(block_class(block_in_channels, out_channels, stride=block_stride, shortcut=shortcut, residual=residual))

    return nn.Sequential(*blocks)


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    return model


class CifarResNet(nn.Module):
    def __init__(self, n, residual=True, shortcut="A", num_classes=10):
        super().__init__()

        self.n = n
        self.residual = residual
        self.shortcut = shortcut

        first_channels, second_channels, third_channels = STAGE_CHANNELS

        self.stem_convolution = nn.Conv2d(
            3, STEM_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stem_normalization = nn.BatchNorm2d(STEM_CHANNELS)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = build_stage(
            STEM_CHANNELS, first_channels, n, 1, residual, shortcut
        )
        self.stage2 = build_stage(
            first_channels, second_channels, n, 2, residual, shortcut
        )
        self.stage3 = build_stage(
            second_channels, third_channels, n, 2, residual, shortcut
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(third_channels, num_classes)

        initialize_weights(self)
        
    @property
    def depth(self):
        return 6 * self.n + 2

    def forward(self, x):
        out = self.stem_convolution(x)
        out = self.stem_normalization(out)
        out = self.relu(out)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

# probably wont use this but I figured id add it either way
class ImageNetResNet(nn.Module):
    def __init__(self, depth, residual=True, shortcut="B", num_classes=1000):
        super().__init__()

        if depth not in IMAGENET_LAYOUTS:
            raise ValueError(
                f"depth must be one of {sorted(IMAGENET_LAYOUTS)}, got {depth}"
            )

        block_class, block_counts = IMAGENET_LAYOUTS[depth]

        self._depth = depth
        self.residual = residual
        self.shortcut = shortcut

        self.stem_convolution = nn.Conv2d(
            3, IMAGENET_STEM_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.stem_normalization = nn.BatchNorm2d(IMAGENET_STEM_CHANNELS)
        self.relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages = []
        in_channels = IMAGENET_STEM_CHANNELS
        for index, (out_channels, block_count) in enumerate(zip(IMAGENET_STAGE_CHANNELS, block_counts)):
            stride = 1 if index == 0 else 2
            stages.append(
                build_stage(
                    in_channels, out_channels, block_count, stride,
                    residual, shortcut, block_class,
                )
            )
            in_channels = out_channels * block_class.expansion
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

        initialize_weights(self)

    def depth(self):
        return self._depth

    def forward(self, x):
        out = self.stem_convolution(x)
        out = self.stem_normalization(out)
        out = self.relu(out)
        out = self.stem_pool(out)

        out = self.stages(out)

        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)


def count_layers(module):
    layers = 0
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            layers += 1
        elif isinstance(child, nn.Conv2d) and "shortcut" not in name:
            layers += 1
    return layers
