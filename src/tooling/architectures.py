#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ~~ IMPORTS ~~
import copy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from warnings import warn

import torch as th
from src.tooling.activations.serf import SERF
from torch import nn
from torch import Tensor
from src.tooling.util.overlap import tensor_pair_from_overlap


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
        for head in self.heads:
            nn.init.normal_(head.weight, 0.0, 0.001)

        self.fxout: nn.Module = copy.deepcopy(activation_fx_module)

        self._switch: bool = True
        self.flip_switch()

    def flip_switch(self) -> None:
        self._switch: bool = not self._switch
        model_reqgrad_(self.heads[int(not self._switch)], False)
        model_reqgrad_(self.heads[int(self._switch)], True)

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

        params += [
            {"params": self.fxout.parameters()}
        ]  # Maybe the activation has learnable parameters. Who knows?

        return params

    def train(self, mode: bool = True):
        # Make sure (only) the active head is trainable/trained
        self.neck.train(mode)
        self.heads[int(self._switch)].train(mode)
        self.fxout.train(mode)
        self.heads[int(not self._switch)].train(False)

    def forward(
        self, x: Tensor, return_both_heads
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        xneck = self.neck(x)
        xpost = self.heads[int(self._switch)](xneck)
        xpost = self.fxout(xpost)

        if return_both_heads:
            with th.no_grad():
                xpre = self.heads[int(not self._switch)](xneck)
                xpre = self.fxout(xpre)

            if self._switch:
                return xpre, xpost
            else:
                return xpost, xpre
        else:
            return xpost


class DoubleTeacher(nn.Module):
    """One Module, Two Teachers: it's a School!"""

    def __init__(
        self,
        in_size: int,
        hid_size: int,
        out_size: int,
        init_features_from: Optional[Tuple[Tensor, Tensor]] = None,
        activation_fx_module=SERF(),
    ) -> None:
        super().__init__()
        
        self.in_size = in_size
        self.out_size = out_size

        self.t1_features = nn.Linear(in_size, hid_size, bias=False)
        self.t2_features = nn.Linear(in_size, hid_size, bias=False)

        if init_features_from is not None:
            self.t1_features.weight.data = init_features_from[0].unsqueeze(0)
            self.t2_features.weight.data = init_features_from[1].unsqueeze(0)

        self.teacher_1 = nn.Sequential(
            self.t1_features,
            copy.deepcopy(activation_fx_module),
            nn.Linear(hid_size, out_size, bias=False),
            copy.deepcopy(activation_fx_module),
        )
        nn.init.normal_(self.teacher_1[2].weight, 0.0, 1.0)

        self.teacher_2 = nn.Sequential(
            self.t2_features,
            copy.deepcopy(activation_fx_module),
            nn.Linear(hid_size, out_size, bias=False),
            copy.deepcopy(activation_fx_module),
        )
        nn.init.normal_(self.teacher_2[2].weight, 0.0, 1.0)

        self.school = nn.ModuleList([self.teacher_1, self.teacher_2])

        self._switch: bool = True
        self.flip_switch()
        self.train(False)

    def train(self, mode: bool = False) -> nn.Module:
        if mode:
            warn(
                f"DoubleTeacher is not trainable, ignoring train(mode={mode})",
                RuntimeWarning,
            )
        for module in self.children():
            module.train(False)
            model_reqgrad_(module, False)
        return self

    def flip_switch(self) -> None:
        self._switch: bool = not self._switch

    def forward(
        self, x: Tensor, return_both_teachers
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
            
    def sample_batch(
        self, n: int, return_both_teachers: bool = False, output_noise_std: float = 0.
    ):
        """Generate iid vectors and associate labels from teacher network."""
        
        X1 = th.normal(0.0, 1.0, size=(n, self.in_size)) # TODO: check this
        if not return_both_teachers:
            y1 = self(X1, return_both_teachers=return_both_teachers)
            y1 += th.randn(y1.size())*output_noise_std
            return X1, y1
        else:
            y1, y2 = self(X1, return_both_teachers=return_both_teachers)
            y1 += th.randn(y1.size())*output_noise_std
            y2 += th.randn(y2.size())*output_noise_std
            return X1, y1, y2
            

def goldt_student(out_size: int) -> TwoHeadStudent:
    return TwoHeadStudent(500, 2, out_size)


def goldt_school(out_size: int) -> DoubleTeacher:
    return DoubleTeacher(500, 1, out_size)


def goldt_school_from_overlap(
    out_size: int, overlap: Union[int, float, Tensor]
) -> DoubleTeacher:
    return DoubleTeacher(
        500, 1, out_size, init_features_from=tensor_pair_from_overlap(overlap, 500)
    )


# TEST
if __name__ == "__main__":
    double_teacher = goldt_school(2)
    student = goldt_student(2)
    overlapped_teachers = goldt_school_from_overlap(1, overlap=0.5)
    dummy_data = th.normal(0.0, 1.0, size=(1, 500))
    double_teacher(dummy_data, False)
    double_teacher(dummy_data, True)
    student(dummy_data, False)
    student(dummy_data, True)
    overlapped_teachers(dummy_data, False)
    overlapped_teachers(dummy_data, True)
