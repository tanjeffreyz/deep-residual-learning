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


class Shortcut(torch.nn.Module):
    def __init__(self, x, option=None):
        """
        Options:
            A - zero-padding
            B - linear projection
        """

        super().__init__()

        assert option in {None, 'A', 'B'}
        self.option = option
        self.input = x

    def forward(self, x):
        if x.shape[1] == self.input.shape[1]:
            return x + self.input
        elif self.option == 'A':
            channels_padded = torch.stack([self.input, torch.zeros_like(self.input)], dim=1)
            dim_padded = torch.nn.functional.pad(x, (0, 0, 0, 0, ), value=0)
            return
        elif self.option == 'B':

            return
        return x
