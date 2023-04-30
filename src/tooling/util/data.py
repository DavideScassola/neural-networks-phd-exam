#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
from collections.abc import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch as th
from torch import nn
from torch import Tensor

# ------------------------------------------------------------------------------


def generate_input(
    input_size: Union[int, Tuple[int, ...]],
    batch_size: int = 1,
    harvest_from_fx: Callable = th.normal,
    harvest_from_args: Optional[Tuple] = None,
    harvest_from_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
):
    """Generate i.i.d. data elements to be fed to a teacher/student network."""

    # Select device
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be either 'cpu' or 'cuda'.")
        if device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

    # Sanity checks + convert to tuple if needed
    if isinstance(input_size, int):
        input_size = (input_size,)
    else:
        if not isinstance(input_size, tuple):
            raise ValueError("input_size must be an int or a tuple of ints.")
        if not all(isinstance(x, int) for x in input_size):
            raise ValueError("input_size must be an int or a tuple of ints.")

    # Build sampler
    if harvest_from_args is None:
        harvest_from_args: Tuple[int, ...] = ()
    if harvest_from_kwargs is None:
        harvest_from_kwargs: dict = {}

    return harvest_from_fx(
        *harvest_from_args,
        **harvest_from_kwargs,
        size=(batch_size, *input_size),
        device=device,
    )


def generate_labelled_input(
    teacher: nn.Module,
    batch_size: int = 1,
    harvest_from_fx: Callable = th.normal,
    harvest_from_args: Optional[Tuple] = None,
    harvest_from_kwargs: Optional[dict] = None,
    labels_noise_fx: Callable = th.normal,
    labels_noise_args: Optional[Tuple] = None,
    labels_noise_kwargs: Optional[dict] = None,
    return_both_teachers: bool = False,
    which_teacher: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
    with th.no_grad():
        # Sanity checks
        if return_both_teachers and which_teacher is not None:
            raise ValueError(
                "return_both_teachers and which_teacher are mutually exclusive."
            )
        if not return_both_teachers and which_teacher is None:
            raise ValueError(
                "Either return_both_teachers or which_teacher must be specified."
            )

        x: Tensor = generate_input(
            input_size=teacher.in_size,
            batch_size=batch_size,
            harvest_from_fx=harvest_from_fx,
            harvest_from_args=harvest_from_args,
            harvest_from_kwargs=harvest_from_kwargs,
            device=device,
        )

        teacher: nn.Module = teacher.to(device)

        # Plan what to do according to user request
        if return_both_teachers or which_teacher != int(teacher._switch):
            actually_return_both_teachers: bool = True
        else:
            actually_return_both_teachers: bool = False

        y: Union[Tensor, Tuple[Tensor, Tensor]] = teacher(
            x, return_both_teachers=actually_return_both_teachers
        )

        # Sample label noise
        if labels_noise_args is None:
            labels_noise_args: Tuple[int, ...] = ()
        if labels_noise_kwargs is None:
            labels_noise_kwargs: dict = {}

        label_noise = labels_noise_fx(
            *labels_noise_args,
            **labels_noise_kwargs,
            size=(2, *y[0].shape) if actually_return_both_teachers else y.shape,
            device=device,
        )

        # Add label noise
        if actually_return_both_teachers:
            y: Tuple[Tensor, Tensor] = y[0] + label_noise[0], y[1] + label_noise[1]
            if return_both_teachers:
                return x, y
            else:
                return x, y[which_teacher]
        else:
            return x, y + label_noise
