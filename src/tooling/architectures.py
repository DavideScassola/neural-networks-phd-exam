#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
import copy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from warnings import warn

import torch as th
from torch import nn
from torch import Tensor

from .nn import ScaledERF
from .util import tensor_pair_from_overlap

# ------------------------------------------------------------------------------


def _model_reqgrad_(model: nn.Module, set_to: Optional[bool]) -> None:
    """Set the requires_grad attribute of all parameters in a model."""
    for parameter in model.parameters():
        parameter.requires_grad = set_to


# ------------------------------------------------------------------------------


class TwoHeadStudent(nn.Module):
    """A multi-head student network."""

    def __init__(
        self,
        in_size: int,
        hid_size: int,
        out_size: int,
        activation_fx_module=ScaledERF(),
    ):
        super().__init__()

        self.in_size = in_size

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

        self.reset_parameters()
        self._switch: bool = True
        self.flip_switch()
        self.train(True)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.neck[0].weight, 0.0, 0.001)
        for head in self.heads:
            nn.init.normal_(head.weight, 0.0, 0.001)

    def flip_switch(self) -> None:
        self._switch: bool = not self._switch
        _model_reqgrad_(self.heads[int(not self._switch)], False)
        _model_reqgrad_(self.heads[int(self._switch)], True)

    def trainable_parameters(
        self, do_scale: bool = True, lr: Optional[float] = None
    ) -> List[Dict[str, Tensor]]:
        params = []
        params += [{"params": self.neck.parameters()}]
        if do_scale:
            if lr is None:
                raise ValueError("lr must be specified if do_scale is True")
            params += [
                {
                    "params": self.heads[int(self._switch)].parameters(),
                    "lr": lr / self.in_size,
                }
            ]

        else:
            params += [{"params": self.heads[int(self._switch)].parameters()}]

        return params

    def train(self, mode: bool = True) -> None:
        # Make sure (only) the active head is trainable/trained
        self.neck.train(mode)
        self.heads[int(self._switch)].train(mode)
        self.heads[int(not self._switch)].train(False)

    def forward(
        self, x: Tensor, return_both_heads: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        xneck = self.neck(x)
        xpost = self.heads[int(self._switch)](xneck)

        if return_both_heads:
            with th.no_grad():
                xpre = self.heads[int(not self._switch)](xneck)

            if self._switch:
                return xpre, xpost
            else:
                return xpost, xpre
        else:
            return xpost


class DoubleTeacher(nn.Module):
    """A set of two teacher networks."""

    def __init__(
        self,
        in_size: int,
        hid_size: int,
        out_size: int,
        init_features_from: Optional[Tuple[Tensor, Tensor]] = None,
        activation_fx_module=ScaledERF(),
    ) -> None:
        super().__init__()

        self.in_size = in_size
        self.initialized_features_from = init_features_from is not None

        self.t1_features = nn.Linear(in_size, hid_size, bias=False)
        self.t2_features = nn.Linear(in_size, hid_size, bias=False)

        if init_features_from is not None:
            self.t1_features.weight.data = init_features_from[0].unsqueeze(0)
            self.t2_features.weight.data = init_features_from[1].unsqueeze(0)

        self.teacher_1 = nn.Sequential(
            self.t1_features,
            copy.deepcopy(activation_fx_module),
            nn.Linear(hid_size, out_size, bias=False)
        )

        self.teacher_2 = nn.Sequential(
            self.t2_features,
            copy.deepcopy(activation_fx_module),
            nn.Linear(hid_size, out_size, bias=False)
        )

        self.school = nn.ModuleList([self.teacher_1, self.teacher_2])

        self.reset_parameters()
        self._switch: bool = True
        self.flip_switch()
        self.train(False)

    def reset_parameters(self) -> None:
        for module in self.school:
            nn.init.normal_(module[2].weight, 0.0, 1.0)
            if not self.initialized_features_from:
                nn.init.normal_(module[0].weight, 0.0, 1.0)

    def train(self, mode: bool = False) -> nn.Module:
        if mode:
            warn(
                f"DoubleTeacher is not trainable, ignoring train(mode={mode})",
                RuntimeWarning,
            )
        for module in self.children():
            module.train(False)
            _model_reqgrad_(module, False)
        return self

    def flip_switch(self) -> None:
        self._switch: bool = not self._switch

    def forward(
        self, x: Tensor, return_both_teachers: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        with th.no_grad():
            xpost = self.school[int(self._switch)](x)

            if return_both_teachers:
                xpre = self.school[int(not self._switch)](x)
                if self._switch:
                    return xpre, xpost
                else:
                    return xpost, xpre
            else:
                return xpost


# ------------------------------------------------------------------------------
def double_teacher_from_overlap(
    in_size: int,
    hid_size: int,
    out_size: int,
    overlap: float,
    activation_fx_module=ScaledERF(),
):
    """Create a DoubleTeacher model from a given overlap."""

    if overlap < 0.0 or overlap > 1.0:
        raise ValueError("overlap must be in [0.0, 1.0]")

    return DoubleTeacher(
        in_size,
        hid_size,
        out_size,
        init_features_from=tensor_pair_from_overlap(overlap, in_size),
        activation_fx_module=activation_fx_module,
    )


# ------------------------------------------------------------------------------
def _test() -> None:
    from .util import tensor_pair_from_overlap

    teacher = DoubleTeacher(
        500,
        1,
        2,
        init_features_from=tensor_pair_from_overlap(
            overlap=0.5, length=500, device="cpu"
        ),
    )
    student = TwoHeadStudent(500, 2, 2)
    dummy_data = th.normal(0.0, 1.0, size=(1, 500))

    teacher(dummy_data, False)
    teacher(dummy_data, True)
    student(dummy_data, False)
    student(dummy_data, True)
