"""
Anatomy of a residual block

            X -----------
            |           |
        weight layer    |
            |           |
        weight layer    |
            |           |
           (+) <---------
            |
           H(X)

This entire block describes the underlying mapping H(X) = F(X) + X where F is the mapping
described by the two weight layers. Rearranging yields F(X) = H(X) - X. This shows that,
instead of directly mapping an input X to an output H(X), the weight layers are responsible
for describing what to change, if anything, about the input X to reach the desired mapping
H(X).

Intuitively, it is easier to modify an existing function than to create a brand new one
from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#############################
#        Components         #
#############################
class Header(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

    def forward(self, x):
        if x.shape[1] != 1 or x.shape[2] != 224 or x.shape[3] != 224:
            raise ValueError(f'Expected input shape (*, 1, 224, 224), got {tuple(x.shape)}')
        return self.model.forward(x)


class Footer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if x.shape[2] != 7 or x.shape[3] != 7:
            raise ValueError(f'Expected input shape (*, *, 7, 7), got {tuple(x.shape)}')
        return self.model.forward(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, in_size, down_sample=False, shortcut=True, option=None):
        super().__init__()

        """
        Padding is calculated as follows:
            (IN_DIM - F + 2P) / S + 1 = OUT_DIM
        F = filter size
        P = padding
        S = stride
        """

        assert option in {None, 'A', 'B'}, f"'{option}' is an invalid option"
        self.in_size = in_size
        self.down_sample = down_sample
        self.shortcut = shortcut
        self.option = option
        if self.down_sample:
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
class ResNet34(nn.Module):
    def __init__(self, residual=True, option=None):
        super().__init__()

        modules = [Header()]
        modules += [DoubleConvBlock(64, 56, shortcut=residual) for _ in range(3)]

        modules.append(DoubleConvBlock(64, 56, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(128, 28, shortcut=residual) for _ in range(3)]

        modules.append(DoubleConvBlock(128, 28, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(256, 14, shortcut=residual) for _ in range(5)]

        modules.append(DoubleConvBlock(256, 14, shortcut=residual, down_sample=True, option=option))
        modules += [DoubleConvBlock(512, 7, shortcut=residual) for _ in range(2)]

        modules += [Footer()]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model.forward(x)


class ResNet18(nn.Module):
    def __init__(self, residual=True, option=None):
        super().__init__()

        self.model = nn.Sequential(
            Header(),
            DoubleConvBlock(64, 56, shortcut=residual),
            DoubleConvBlock(64, 56, shortcut=residual),
            DoubleConvBlock(64, 56, shortcut=residual, down_sample=True, option=option),
            DoubleConvBlock(128, 28, shortcut=residual),
            DoubleConvBlock(128, 28, shortcut=residual, down_sample=True, option=option),
            DoubleConvBlock(256, 14, shortcut=residual),
            DoubleConvBlock(256, 14, shortcut=residual, down_sample=True, option=option),
            DoubleConvBlock(512, 7, shortcut=residual),
            Footer()
        )

    def forward(self, x):
        return self.model.forward(x)
