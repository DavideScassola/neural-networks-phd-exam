#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
from typing import Optional
from typing import Tuple
from typing import Union

import torch as th
from scipy.stats import ortho_group as spsog
from torch import cos
from torch import pi
from torch import sin
from torch import Tensor

# ------------------------------------------------------------------------------


def tensor_pair_from_angle(
    angle: Union[int, float, Tensor], length: int, device: Optional[str] = None
) -> Tuple[Tensor, Tensor]:
    """Generate a pair of normalized torch.Tensors, along directions forming given angle in radians."""

    # Select device
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be either 'cpu' or 'cuda'.")
        if device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

    # Convert to float tensor
    angle: Tensor = (angle if isinstance(angle, Tensor) else th.tensor(angle)).to(
        device
    )

    # Sanity checks
    if angle.shape != ():
        raise ValueError("angle must be a scalar (0-dimensional) torch.Tensor.")
    if angle < 0 or angle >= 2 * pi:
        raise ValueError("angle must be between 0 and 2*pi.")

    # Generator vectors
    vgen1: Tensor = th.tensor([0.0, 1.0], device=device)
    vgen2: Tensor = th.tensor([sin(angle), cos(angle)], device=device)

    # Haar-harvested O(N) vectors
    rgen: Tensor = th.from_numpy(spsog(length).rvs())[:, 0:2].float().to(device)

    # Compute the pair
    v1: Tensor = rgen @ vgen1
    v2: Tensor = rgen @ vgen2

    # Return the pair
    return v1, v2


def tensor_pair_from_overlap(
    overlap: Union[int, float, Tensor], length: int, device: Optional[str] = None
) -> Tuple[Tensor, Tensor]:
    """Generate a pair of normalized torch.Tensors, with given cosine overlap."""

    # Overlap conversion to tensor
    if not isinstance(overlap, Tensor):
        overlap: Tensor = th.tensor(overlap)

    return tensor_pair_from_angle(th.acos(overlap), length, device)


# ------------------------------------------------------------------------------


def _test() -> None:
    otest: Tensor = th.rand(1)[0]
    vtest_1, vtest_2 = tensor_pair_from_overlap(overlap=otest, length=10, device="cpu")
    assert th.allclose(vtest_1.dot(vtest_2), otest)
