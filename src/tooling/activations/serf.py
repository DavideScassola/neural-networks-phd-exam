#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ IMPORTS ~~
import math

import torch
from torch import Tensor


@torch.jit.script
def fserf(x: Tensor) -> Tensor:
    """Functional scaled ERror Function."""
    return torch.erf(x / math.sqrt(2.0))  # type: ignore


class SERF(torch.nn.Module):
    """Scaled ERror Function."""

    def __init__(self) -> None:
        super(SERF, self).__init__()

    def forward(self, x: Tensor) -> Tensor:  # Do not make static!
        return fserf(x)


# TEST
if __name__ == "__main__":
    ytest = fserf(torch.linspace(-10, 10, 1000))
