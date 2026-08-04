from torch import nn
import torch.nn.functional as F

VALID_SHORTCUT_OPTIONS = ("A", "B", "C")


class ZeroPadShortcut(nn.Module):
    # option a is subsample spatially, pad the new channels with zeros
    # ts the best one and what we gonna be doing

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError(
                f"cannot pad {in_channels} channels down to {out_channels}"
            )

        self.stride = stride
        self.padding_channels = out_channels - in_channels

    def forward(self, x):
        subsampled = x[:, :, ::self.stride, ::self.stride]
        return F.pad(subsampled, (0, 0, 0, 0, 0, self.padding_channels))


class ProjectionShortcut(nn.Module):
    # options b and c, a 1x1 convolution, with a batch norm after it

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.normalization(self.convolution(x))


def make_shortcut(in_channels, out_channels, stride, option="A"):
    if option not in VALID_SHORTCUT_OPTIONS:
        raise ValueError(
            f"shortcut option must be one of {VALID_SHORTCUT_OPTIONS}, got {option!r}"
        )

    shape_changes = stride != 1 or in_channels != out_channels

    if option == "C":
        return ProjectionShortcut(in_channels, out_channels, stride)

    if not shape_changes:
        return None

    if option == "A":
        return ZeroPadShortcut(in_channels, out_channels, stride)

    return ProjectionShortcut(in_channels, out_channels, stride)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None, residual=True):
        super().__init__()

        self.residual = residual
        self.shortcut = shortcut

        expanded_channels = out_channels * self.expansion

        self.convolution1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.normalization1 = nn.BatchNorm2d(out_channels)
        self.convolution2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.normalization2 = nn.BatchNorm2d(out_channels)

        self.convolution3 = nn.Conv2d(
            out_channels, expanded_channels, kernel_size=1, bias=False
        )
        self.normalization3 = nn.BatchNorm2d(expanded_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.normalization1(self.convolution1(x)))
        out = self.relu(self.normalization2(self.convolution2(out)))
        out = self.normalization3(self.convolution3(out))

        if self.residual:
            if self.shortcut is not None:
                identity = self.shortcut(x)
            out = out + identity

        return self.relu(out)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, shortcut=None, residual=True):
        super().__init__()

        self.residual = residual
        self.shortcut = shortcut
        self.convolution1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, 
                                      kernel_size=3, stride=stride, padding=1, bias=False)
        self.normalization1 = nn.BatchNorm2d(out_channels)
        self.convolution2 = nn.Conv2d(in_channels=out_channels, out_channels= out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.normalization2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    # here is the actual block being built and we go through all of the steps ts tuf
    def forward(self, x):
        identity = x
        out = self.convolution1(x)
        out = self.normalization1(out)
        out = self.relu(out)
        out = self.convolution2(out)
        out = self.normalization2(out)
        
        if self.residual:
            if self.shortcut is not None:
                identity = self.shortcut(x)
            out = out + identity
        out = self.relu(out)
        return out