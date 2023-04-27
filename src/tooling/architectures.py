#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ IMPORTS ~~
import copy
from typing import Optional
from typing import Tuple
from typing import Union

import torch as th
from torch import nn
from torch import Tensor

from ..tooling.activations.serf import SERF
from ..tooling.util.overlap import tensor_pair_from_overlap


def model_reqgrad_(model: nn.Module, set_to: Optional[bool]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = set_to


class TwoHeadStudent(nn.Module):
    """A multi-head student network."""

    def __init__(
        self,
        in_size: int,
        hid_size: int,
        out_size: int,
        activation_fx_module=SERF(),
    ):
        super().__init__()

        self.neck: nn.Module = nn.Sequential(
            nn.Linear(in_size, hid_size, bias=False),
            copy.deepcopy(activation_fx_module),
        )
        self.heads = nn.ModuleList(
            [
                nn.Linear(hid_size, out_size, bias=False),
                nn.Linear(hid_size, out_size, bias=False),
            ]
        )
        self.fxout: nn.Module = copy.deepcopy(activation_fx_module)

        self._switch: bool = True
        self.flip_switch()

    def flip_switch(self) -> None:
        self._switch: bool = not self._switch
        model_reqgrad_(self.heads[not self._switch], False)

    def forward(
        self, x: Tensor, return_both_heads
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        xneck = self.neck(x)
        xpost = self.heads[self._switch](xneck)
        xpost = self.fxout(xpost)

        if return_both_heads:
            with th.no_grad():
                xpre = self.heads[not self._switch](xneck)
                xpre = self.fxout(xpre)

            if self._switch:
                return xpre, xpost
            else:
                return xpost, xpre
        else:
            return xpost


class DoubleTeacher(nn.Module):
    """One Module, Two Teachers!"""

    def __init__(
        self,
        in_size: int,
        hid_size: int,
        out_size: int,
        init_features_from: Optional[Tuple[Tensor, Tensor]] = None,
        activation_fx_module=SERF(),
    ):
        super().__init__()

        self.t1_features = nn.Linear(in_size, hid_size, bias=False)
        self.t2_features = nn.Linear(in_size, hid_size, bias=False)

        if init_features_from is not None:
            self.t1_features.weight.data = init_features_from[0]
            self.t2_features.weight.data = init_features_from[1]

        self.teacher_1 = nn.Sequential(
            self.t1_features,
            copy.deepcopy(activation_fx_module),
            nn.Linear(hid_size, out_size, bias=False),
            copy.deepcopy(activation_fx_module),
        )

        self.teacher_2 = nn.Sequential(
            self.t2_features,
            copy.deepcopy(activation_fx_module),
            nn.Linear(hid_size, out_size, bias=False),
            copy.deepcopy(activation_fx_module),
        )
        self.school = nn.ModuleList([self.teacher_1, self.teacher_2])

        self._switch: bool = True
        self.flip_switch()

    def flip_switch(self) -> None:
        self._switch: bool = not self._switch
        model_reqgrad_(self.heads[not self._switch], False)

    def forward(
        self, x: Tensor, return_both_teachers
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        xpost = self.school[self._switch](x)

        if return_both_teachers:
            xpre = self.school[not self._switch](x)
            if self._switch:
                return xpre, xpost
            else:
                return xpost, xpre
        else:
            return xpost


def goldt_student(out_size: int):
    return TwoHeadStudent(500, 2, out_size)


def goldt_school(out_size: int):
    return DoubleTeacher(500, 1, out_size)


def goldt_school_from_overlap(out_size: int, overlap: Tensor):
    return DoubleTeacher(
        500, 1, out_size, init_features_from=tensor_pair_from_overlap(overlap, 500)
    )
