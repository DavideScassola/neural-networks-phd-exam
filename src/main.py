#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
from copy import deepcopy
import os
from typing import Tuple

import numpy as np
import torch as th
from tooling.architectures import double_teacher_from_overlap
from tooling.architectures import TwoHeadStudent
from tooling.util import generate_labelled_input
from tooling.util import train_student_head_otf
from tooling.util import training_trace_fig
from tooling.util import TrainingTrace
from tqdm.auto import tqdm


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
EXPERIMENT_SEED = 2054 # np.random.randint(0, 2**32 - 1)
OVERLAP_SEED = None
RUN_OVERLAPS = [float(i) for i in np.linspace(0.0, 1.0, 10)]

IN_SIZE = 500
OUT_SIZE = 1
TEACH_HID = 1
STUD_HID = 2
INPUT_MEAN = 0.0
INPUT_STD = 1.0
LABELS_NOISE_STD = 0.01

BATCH_SIZE = 64
TRAIN_STEPS = 8192
EVAL_EVERY = 20

OPTIM = th.optim.SGD
LR = 1.0
CRITERION = th.nn.MSELoss()

DEVICE = "cpu"

# ------------------------------------------------------------------------------


def continual_learning_run(overlap: float):
    # Set seeds
    np.random.seed(seed=EXPERIMENT_SEED)
    th.manual_seed(seed=EXPERIMENT_SEED)

    # Instantiate teachers
    teacher = double_teacher_from_overlap(
        IN_SIZE,
        TEACH_HID,
        OUT_SIZE,
        overlap=overlap,
        seedwith=OVERLAP_SEED,
    ).to(DEVICE)

    # Generate test set
    x_test, y_test = generate_labelled_input(
        teacher,
        BATCH_SIZE,
        harvest_from_fx=th.normal,
        harvest_from_args=(INPUT_MEAN, INPUT_STD),
        labels_noise_fx=th.normal,
        labels_noise_args=(0.0, LABELS_NOISE_STD),
        return_both_teachers=True,
        device=DEVICE,
    )

    # Instantiate student
    student = TwoHeadStudent(IN_SIZE, STUD_HID, OUT_SIZE).to(DEVICE)

    # Train student on task 1
    train_trace_1 = train_student_head_otf(
        student,
        teacher,
        0,
        TRAIN_STEPS,
        BATCH_SIZE,
        OPTIM,
        LR,
        CRITERION,
        INPUT_MEAN,
        INPUT_STD,
        LABELS_NOISE_STD,
        DEVICE,
        eval_every=EVAL_EVERY,
        eval_on_x=x_test,
        eval_on_y=y_test,
        print_overlap=overlap,
    )

    # Flip student's heads and switch task
    teacher.flip_switch()
    student.flip_switch()

    # Train student on task 2
    train_trace_2 = train_student_head_otf(
        student,
        teacher,
        1,
        TRAIN_STEPS,
        BATCH_SIZE,
        OPTIM,
        LR,
        CRITERION,
        INPUT_MEAN,
        INPUT_STD,
        LABELS_NOISE_STD,
        DEVICE,
        eval_every=EVAL_EVERY,
        eval_on_x=x_test,
        eval_on_y=y_test,
        resume_it_from=TRAIN_STEPS,
        print_overlap=overlap,
    )

    xtrace = train_trace_1[0] + train_trace_2[0]
    ytrace_task_1 = train_trace_1[1][0] + train_trace_2[1][0]
    ytrace_task_2 = train_trace_1[1][1] + train_trace_2[1][1]

    return TrainingTrace(x=xtrace, y=(ytrace_task_1, ytrace_task_2), overlap=overlap)


def continual_learning_experiment(overlaps: Tuple[float, ...]):
    traces = []
    for overlap in tqdm(overlaps, leave=True, desc="Testing overlaps"):  # type: ignore
        traces.append(continual_learning_run(overlap))  # type: ignore
    return traces


def main_runner():
    cle = continual_learning_experiment(overlaps=RUN_OVERLAPS)

    if not os.path.exists("../plots"):
        os.makedirs("../plots")

    fig = training_trace_fig(cle)
    fig.write_image("../plots/tasks.png", scale=2.0)


if __name__ == "__main__":
    main_runner()
