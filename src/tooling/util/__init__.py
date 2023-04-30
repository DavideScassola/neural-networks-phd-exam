#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
from .data import generate_input
from .data import generate_labelled_input
from .experiment import TrainingTrace
from .loops import test_student
from .loops import train_student_head_otf
from .loops import train_student_one_step
from .overlap import tensor_pair_from_angle
from .overlap import tensor_pair_from_overlap
from .plot import training_trace_fig

del overlap
del data
del loops
del experiment
# ------------------------------------------------------------------------------
