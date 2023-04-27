#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ IMPORTS ~~
from typing import Optional
from typing import Tuple
from typing import Union

import torch as th
from scipy.stats import ortho_group as spog
from torch import cos
from torch import pi
from torch import sin
from torch import Tensor

# ~~ FUNCTIONS ~~


def tensor_pair_from_angle(
    angle: Union[int, float, Tensor], length: int, device: Optional[str] = None
) -> Tuple[Tensor, Tensor]:
    """Returns a pair of normalized torch.Tensors whose directions form given angle."""

    # Select device
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be either 'cpu' or 'cuda'.")
        if device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

    # Convert to float tensor
    angle = (angle if isinstance(angle, Tensor) else th.tensor(angle)).to(device)

    # Sanity checks
    if angle.shape != ():
        raise ValueError("angle must be a scalar (0-dimensional) torch.Tensor.")
    if angle < 0 or angle > 2 * pi:
        raise ValueError("angle must be between 0 and 2*pi.")

    # Generator vectors
    vgen1 = th.tensor([0.0, 1.0], device=device)
    vgen2 = th.tensor([sin(angle), cos(angle)], device=device)

    # Haar-harvested O(N) vectors
    rgen = th.from_numpy(spog(length).rvs())[:, 0:2].float().to(device)

    # Compute the pair
    v1 = rgen @ vgen1
    v2 = rgen @ vgen2

    # Return the pair
    return v1, v2


def tensor_pair_from_overlap(
    overlap: Union[int, float, Tensor], length: int, device: Optional[str] = None
) -> Tuple[Tensor, Tensor]:
    """Returns a pair of normalized torch.Tensors with given overlap."""
    return tensor_pair_from_angle(th.acos(overlap), length, device)


# TEST
if __name__ == "__main__":
    otest = th.rand(1)[0]
    vtest_1, vtest_2 = tensor_pair_from_overlap(overlap=otest, length=10, device="cpu")
    assert th.allclose(vtest_1.dot(vtest_2), otest)
