#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
import math

import torch
from torch import nn
from torch import Tensor

# ------------------------------------------------------------------------------


@torch.jit.script
def functional_serf(x: Tensor) -> Tensor:
    """Scaled ERror Function, torch.functional version."""
    return torch.erf(x / math.sqrt(2.0))  # type: ignore


class ScaledERF(torch.nn.Module):
    """Scaled ERror Function, torch.nn.Module version."""

    def __init__(self) -> None:
        super(ScaledERF, self).__init__()

    def forward(self, x: Tensor) -> Tensor:  # Do not make static!
        return functional_serf(x)


# ------------------------------------------------------------------------------


def _test() -> None:
    xtest: Tensor = torch.linspace(-10, 10, 1000, device="cpu")
    _: Tensor = functional_serf(xtest)
    serftest: nn.Module = ScaledERF()
    _: Tensor = serftest(xtest)
