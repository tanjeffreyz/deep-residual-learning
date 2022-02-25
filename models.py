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


class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, shortcut=True, option=None):
        super().__init__()

        """
        Padding is calculated as follows:
            (IN - F + 2P) / S + 1 = OUT
        IN = input dimension
        F = filter size
        P = single-side padding
        S = stride
        """

        assert option in {None, 'A', 'B'}, f"'{option}' is an invalid option"
        self.in_size = in_size
        self.out_size = out_size
        self.shortcut = shortcut
        self.option = option
        if in_size == out_size:
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1),
                nn.Conv2d(out_size, out_size, 3, padding=1)
            )
            self.down_sample = False
        elif in_size * 2 == out_size:
            if self.shortcut:
                assert option is not None, 'Must specify an option when downscaling with a shortcut'
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, stride=2, padding=1),
                nn.Conv2d(out_size, out_size, 3, padding=1)
            )
            self.down_sample = True
        else:
            raise ValueError('Either IN_SIZE == OUT_SIZE or IN_SIZE * 2 == OUT_SIZE must be True')

    def forward(self, x):
        if not self.shortcut:               # No residual
            return self.model.forward(x)
        elif self.down_sample:
            if self.option == 'A':          # Zero padding
                return self.model.forward(x)
            elif self.option == 'B':        # Linear projection
                return self.model.forward(x)
        else:                               # Simple residual
            return self.model.forward(x) + x


#############################
#       Architectures       #
#############################
class ResNet34(nn.Module):
    def __init__(self, residual=True, option=None):
        super().__init__()

        modules = [
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]

        modules += [DoubleConvBlock(64, 64, shortcut=residual, option=option) for _ in range(3)]

        modules.append(DoubleConvBlock(64, 128, shortcut=residual, option=option))
        modules += [DoubleConvBlock(128, 128, shortcut=residual, option=option) for _ in range(3)]

        modules.append(DoubleConvBlock(128, 256, shortcut=residual, option=option))
        modules += [DoubleConvBlock(256, 256, shortcut=residual, option=option) for _ in range(5)]

        modules.append(DoubleConvBlock(256, 512, shortcut=residual, option=option))
        modules += [DoubleConvBlock(512, 512, shortcut=residual, option=option) for _ in range(2)]

        modules += [
            nn.AvgPool2d(7),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 1000),
            nn.Softmax(dim=1)
        ]

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model.forward(x)


#####################
#       Script      #
#####################
if __name__ == '__main__':
    def sanity_check(model, batch_size=256):
        name = type(model).__name__
        print(f'\n[~] Checking {name}:')
        test_input = torch.rand(batch_size, 1, 224, 224)
        result = model.forward(test_input)
        shape = result.shape
        if len(shape) == 2 and shape[0] == batch_size and shape[1] == 1000:
            print(f' -  {name} produced correct shape ({batch_size}, 1000)')
        else:
            print(f' !  {name} produced shape {tuple(shape)} when ({batch_size}, 1000) was expected')

    sanity_check(ResNet34(residual=False))      # Plain 34-Layer
    sanity_check(ResNet34(option='A'))
