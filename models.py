"""Various ResNet architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    def f1(module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')

    def f2(module):
        if isinstance(module, DoubleConvBlock):
            ds = module.conv_downsample
            if ds:
                ds.weight.data.fill_(1 / module.in_channels)
                ds.bias.data.fill_(0)

    m.apply(f1)         # Applies weight initialization recursively to all submodules
    m.apply(f2)


#############################
#        Components         #
#############################
class Footer(nn.Module):
    def __init__(self, in_channels, in_size, out_labels):
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.model = nn.Sequential(
            nn.AvgPool2d(in_size),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, out_labels)
        )

    def forward(self, x):
        if x.shape[2] != self.in_size or x.shape[3] != self.in_size:
            raise ValueError(f'Expected input shape (*, {self.in_channels}, {self.in_size}, {self.in_size}), '
                             f'got {tuple(x.shape)}')
        return self.model.forward(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, in_size, shortcut=True, down_sample=False, option=None):
        super().__init__()

        """
        Padding is calculated as follows:
            (IN_DIM - F + 2P) / S + 1 = OUT_DIM
        F = filter size
        P = padding
        S = stride
        """

        assert option in {None, 'A', 'B'}, f"'{option}' is an invalid option"
        self.in_channels = in_channels
        self.in_size = in_size
        self.down_sample = down_sample
        self.shortcut = shortcut
        self.option = option
        self.conv_downsample = None
        if self.down_sample:
            if shortcut:
                assert option is not None, 'Must specify either option A or B when ' \
                                           'downsampling with a residual shortcut'
            out_channels = in_channels * 2
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
            self.conv_downsample = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        if not self.shortcut:               # No residual
            result = self.model.forward(x)
        elif not self.down_sample:          # Simple residual
            result = self.model.forward(x) + x
        elif self.option == 'A':            # Zero padding
            y = self.model.forward(x)
            x = F.max_pool2d(x, 1, 2)
            padded = torch.cat((x, torch.zeros_like(x)), dim=1)
            result = y + padded
        else:                               # Linear projection
            result = self.model.forward(x) + self.conv_downsample.forward(x)
        return F.relu(result)


#############################
#       Architectures       #
#############################
def common_str(obj):
    strings = [
        obj.__class__.__name__,
        str(obj.locals['n']),
        'R' if obj.locals['residual'] else 'P'
    ]
    option = obj.locals['option']
    if option is not None:
        strings.append(option)
    return '-'.join(strings)


class CifarResNet(nn.Module):
    def __init__(self, n, residual=True, option=None):
        self.locals = locals()
        super().__init__()

        num_layers = {20, 32, 44, 56, 110}
        assert n in num_layers, f'N must be in {list(sorted(num_layers))}'
        k = (n - 2) // 6

        modules = [nn.Conv2d(3, 16, 3, padding=1)]
        modules += [DoubleConvBlock(16, 32, shortcut=residual) for _ in range(k)]

        modules.append(DoubleConvBlock(16, 32, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(32, 16, shortcut=residual) for _ in range(k - 1)]

        modules.append(DoubleConvBlock(32, 16, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(64, 8, shortcut=residual) for _ in range(k - 1)]

        modules.append(Footer(64, 8, 10))
        self.model = nn.Sequential(*modules)
        init_weights(self)

    def forward(self, x):
        return self.model.forward(x)

    @staticmethod
    def transform(x):
        return x - torch.mean(x, (1, 2), keepdim=True)

    def __str__(self):
        return common_str(self)


class ImageNetResNet(nn.Module):
    def __init__(self, n, residual=True, option=None):
        self.locals = locals()
        super().__init__()

        assert n in {18, 34}, 'N must either be 18 or 34'
        if n == 18:
            layers = (2, 1, 1, 1)
        else:
            layers = (3, 3, 5, 2)

        modules = [
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]
        modules += [DoubleConvBlock(64, 56, shortcut=residual) for _ in range(layers[0])]

        modules.append(DoubleConvBlock(64, 56, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(128, 28, shortcut=residual) for _ in range(layers[1])]

        modules.append(DoubleConvBlock(128, 28, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(256, 14, shortcut=residual) for _ in range(layers[2])]

        modules.append(DoubleConvBlock(256, 14, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(512, 7, shortcut=residual) for _ in range(layers[3])]

        modules += [Footer(512, 7, 1000)]
        self.model = nn.Sequential(*modules)
        init_weights(self)

    def forward(self, x):
        return self.model.forward(x)

    def __str__(self):
        return common_str(self)
