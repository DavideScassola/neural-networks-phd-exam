#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Tuple


# ------------------------------------------------------------------------------
@dataclass
class TrainingTrace:
    overlap: float
    x: Tuple
    y: Tuple[Tuple, Tuple]

    def get_trace_task(self, task: int):
        return self.x, self.y[task]


# ------------------------------------------------------------------------------
