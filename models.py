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


class Shortcut(nn.Module):
    def __init__(self, x, option=None):
        super().__init__()


        self.option = option
        self.input = x

    def forward(self, x):
        if x.shape[1] == self.input.shape[1]:       # Identity
            return self.input + x
        elif self.option == 'A':                    # Zero padding
            channels_padded = torch.stack([self.input, torch.zeros_like(self.input)], dim=1)
            diff = self.input.shape[2] - x.shape[2]
            pad1 = diff / 2
            pad2 = self.input.shape[2] - pad1
            dim_padded = nn.functional.pad(x, (0, 0, 0, 0, pad1, pad2, pad1, pad2), value=0)
            return channels_padded + dim_padded
        elif self.option == 'B':                    # Linear projection

            return x    # TODO: linear projection
        return x


class SimpleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, option=None):
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
        self.option = option
        if in_size == out_size:
            assert option is None, 'Cannot specify an option when not downsampling'
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1),
                nn.Conv2d(out_size, out_size, 3, padding=1)
            )
        elif in_size * 2 == out_size:
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, stride=2, padding=1),
                nn.Conv2d(out_size, out_size, 3, padding=1)
            )
        else:
            raise ValueError('Either IN_SIZE == OUT_SIZE or IN_SIZE * 2 == OUT_SIZE must be True.')

    def forward(self, x):
        if self.option is None:         # Identity
            return self.model.forward(x)
        elif self.option == 'A':        # Zero padding
            return x
        elif self.option == 'B':        # Linear projection
            return x


class Plain34Layer(nn.Module):
    def __init__(self):
        super().__init__()

        modules = [
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]
        modules += [SimpleConvBlock(64, 64)] * 3

        modules.append(SimpleConvBlock(64, 128))
        modules += [SimpleConvBlock(128, 128)] * 3

        modules.append(SimpleConvBlock(128, 256))
        modules += [SimpleConvBlock(256, 256)] * 5

        modules.append(SimpleConvBlock(256, 512))
        modules += [SimpleConvBlock(512, 512)] * 2

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model.forward(x)


if __name__ == '__main__':
    model = Plain34Layer()
    test_input = torch.rand((256, 1, 224, 224))
    result = model.forward(test_input)
    print(result.shape)
