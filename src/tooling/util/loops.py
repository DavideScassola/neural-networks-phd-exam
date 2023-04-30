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
from torch.optim import Optimizer
from tqdm.auto import trange

from .data import generate_labelled_input

# ------------------------------------------------------------------------------


def train_student_one_step(
    x: Tensor, y: Tensor, student: nn.Module, optimizer: Optimizer, criterion: Callable
) -> float:
    student.train()
    optimizer.zero_grad()
    loss: Tensor = criterion(student(x, return_both_heads=False), y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test_student(
    x: Tensor, y: Union[Tensor, Tuple[Tensor, Tensor]], student: nn.Module, criterion
) -> Union[float, Tuple[float, float]]:
    with th.no_grad():
        student.eval()

        if isinstance(y, tuple):
            y_hat = student(x, return_both_heads=True)
            loss = criterion(y_hat[0], y[0]).item(), criterion(y_hat[1], y[1]).item()
        else:
            y_hat = student(x, return_both_heads=False)
            loss = criterion(y_hat, y).item()

        return loss


def train_student_head_otf(
    student: nn.Module,
    teacher: nn.Module,
    which_head: int,
    train_steps: int,
    batch_size: int,
    optimizer_fx: Callable,
    lr: float,
    criterion: Callable,
    input_mean: float,
    input_std: float,
    labels_noise_std: float,
    device: Optional[str] = None,
    eval_every: Optional[int] = None,
    eval_on_x: Optional[Tensor] = None,
    eval_on_y: Optional[Tensor] = None,
    resume_it_from: Optional[int] = 0,
    print_overlap: Optional[float] = None,
):
    optim: Optimizer = optimizer_fx(
        student.trainable_parameters(do_scale=True, lr=lr), lr=lr
    )

    itseries = []
    lfseries = [[], []]
    print_overlap = "" if print_overlap is None else f"{print_overlap}"
    for it in trange(train_steps, leave=False, desc=f"Training for overlap {print_overlap}, student head #{which_head+1}"):  # type: ignore
        x, y = generate_labelled_input(
            teacher,
            batch_size,
            harvest_from_fx=th.normal,
            harvest_from_args=(input_mean, input_std),
            labels_noise_fx=th.normal,
            labels_noise_args=(0.0, labels_noise_std),
            which_teacher=which_head,
            device=device,
        )

        train_student_one_step(x, y, student, optim, criterion)

        if eval_every is not None and it % eval_every == 0:
            if eval_on_x is not None and eval_on_y is not None:
                eval_lf = test_student(eval_on_x, eval_on_y, student, criterion)
            else:
                eval_lf = test_student(x, y, student, criterion)

            itseries.append(it + resume_it_from)
            if isinstance(eval_lf, tuple):
                lfseries[0].append(eval_lf[0])
                lfseries[1].append(eval_lf[1])
            else:
                lfseries[0].append(eval_lf)

    if len(itseries) > 0:
        if len(lfseries[1]) > 0:
            return itseries, lfseries
        else:
            return itseries, lfseries[0]
