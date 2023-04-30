#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Tuple
import os

import plotly.graph_objects as go
import torch as th
from plotly.colors import qualitative as pxcql
from plotly.subplots import make_subplots
from tooling.architectures import double_teacher_from_overlap
from tooling.architectures import TwoHeadStudent
from tooling.util import generate_labelled_input
from tooling.util import train_student_head_otf
from tqdm.auto import tqdm


# ------------------------------------------------------------------------------
@dataclass
class TrainingTrace:
    overlap: float
    x: Tuple
    y: Tuple[Tuple, Tuple]

    def get_trace_task(self, task: int):
        return self.x, self.y[task]


# ------------------------------------------------------------------------------
IN_SIZE = 500
OUT_SIZE = 1
TEACH_HID = 1
STUD_HID = 2
INPUT_MEAN = 0.0
INPUT_STD = 1.0
LABELS_NOISE_STD = 0.01

BATCH_SIZE = 64
TEST_SIZE = 2048
TRAIN_STEPS = 5000
EVAL_EVERY = 100

OPTIM = th.optim.SGD
LR = 1.0
CRITERION = th.nn.MSELoss()

DEVICE = "cpu"

# ------------------------------------------------------------------------------


def continual_learning_run(overlap: float):
    # Instantiate teachers
    teacher = double_teacher_from_overlap(
        IN_SIZE, TEACH_HID, OUT_SIZE, overlap=overlap
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
    cle = continual_learning_experiment(overlaps=(0.0, 0.25, 0.5, 0.75, 1.0))

    task_x_traces = cle[0].x
    task_y_overlaps = tuple([elem.overlap for elem in cle])
    task_1_traces = tuple([elem.get_trace_task(0)[1] for elem in cle])
    task_2_traces = tuple([elem.get_trace_task(1)[1] for elem in cle])

    # ------------------------------------------------------------------------------

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Task 1", "Task 2"), shared_yaxes=True
    )
    for i in range(len(task_y_overlaps)):
        fig.add_trace(
            go.Scatter(
                x=task_x_traces,
                y=task_1_traces[i],
                mode="lines",
                name=f"overlap {task_y_overlaps[i]}",
                line=dict(color=pxcql.Vivid[i]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=task_x_traces,
                y=task_2_traces[i],
                mode="lines",
                name=f"overlap {task_y_overlaps[i]}",
                line=dict(color=pxcql.Vivid[i]),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    for j in (1, 2):
        fig.update_xaxes(title_text="step", row=1, col=j)
    fig.update_yaxes(title_text="", type="log", row=1, col=2)
    fig.update_yaxes(title_text="loss", type="log", row=1, col=1)

    if not os.path.exists('../plots'):
        os.makedirs('../plots')

    fig.write_image("../plots/tasks.png", scale=2.0)


if __name__ == "__main__":
    main_runner()
