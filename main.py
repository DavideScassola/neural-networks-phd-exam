#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

from src.tooling.architectures import DoubleTeacher
from src.tooling.architectures import TwoHeadStudent
from src.tooling.architectures import tensor_pair_from_overlap
import matplotlib.pyplot as plt
import tqdm
import copy
import numpy as np

INPUT_DIMENSION = 500
TEACHER_HIDDEN_UNITS = 1
STUDENT_HIDDEN_UNITS = 2
OUTPUT_DIMENSION = 1
TRAIN_PROPORTION = 0.8
BATCH_SIZE = 64
TEST_SIZE = 1000
LABELS_NOISE_STD = 0.01
FIRST_SGD_ITERATIONS = int(5e3)
SECOND_SGD_ITERATIONS = FIRST_SGD_ITERATIONS


def train(
    *,
    student: TwoHeadStudent,
    optimizer: torch.optim.Optimizer,
    batches: int,
    teacher: DoubleTeacher,
    test_set: tuple,
    number_of_tests: int = 100,
):
    """Training for the student network."""

    loss_fn = nn.MSELoss()
    student.train()

    # Metrics
    test_losses = {"index": [], "first_task": [], "second_task": []}

    test_evaluation_frequency = batches // number_of_tests

    for i in tqdm.tqdm(range(batches)):
        x_batch, y_batch = teacher.sample_batch(
            n=BATCH_SIZE, output_noise_std=LABELS_NOISE_STD
        )
        optimizer.zero_grad()

        loss = loss_fn(student(x_batch, return_both_heads=False), y_batch)
        loss.backward()
        optimizer.step()

        if (i % test_evaluation_frequency) == 0:
            test_loss1, test_loss2 = evaluate_on_test(
                student=student, test_set=test_set
            )
            test_losses["index"].append(i)
            test_losses["first_task"].append(test_loss1)
            test_losses["second_task"].append(test_loss2)

    return test_losses


def evaluate_on_test(
    *,
    student: TwoHeadStudent,
    test_set: tuple,
):
    """Evaluate the student network on the teacher labels."""

    loss_fn = nn.MSELoss()
    x_test, y_test1, y_test2 = test_set

    with torch.no_grad():
        student.train(False)
        out1, out2 = student(x_test, return_both_heads=True)
        loss1 = loss_fn(out1, y_test1).item()
        loss2 = loss_fn(out2, y_test2).item()
        return loss1, loss2


def train_student(
    *, student: TwoHeadStudent, double_teacher: DoubleTeacher, test_set: tuple
):
    optimizer = torch.optim.SGD(student.trainable_parameters(lr=1), lr=1)

    test_losses = train(
        student=student,
        optimizer=optimizer,
        teacher=double_teacher,
        test_set=test_set,
        batches=FIRST_SGD_ITERATIONS,
    )

    return test_losses


def overlapped_double_teacher(
    *, in_size: int, out_size: int, hid_size: int, overlap
) -> DoubleTeacher:
    return DoubleTeacher(
        in_size,
        hid_size,
        out_size,
        init_features_from=tensor_pair_from_overlap(overlap, in_size),
    )


def apply_common_sense_overlap(x1: torch.Tensor, x2: torch.Tensor, *, overlap: float):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    n = torch.numel(x1)
    sample_indexes = torch.randperm(n)[: int(overlap * n)]
    x1.flatten()[sample_indexes] = x2.flatten()[sample_indexes]


def common_sense_overlapped_double_teacher(
    *, in_size: int, out_size: int, hid_size: int, overlap: float
) -> DoubleTeacher:
    dt = DoubleTeacher(
        in_size,
        hid_size,
        out_size,
    )
    apply_common_sense_overlap(
        dt.t2_features.weight.data, dt.t1_features.weight.data, overlap=overlap
    )
    return dt


def continual_learning_experiment(*, overlap=0.0, student: TwoHeadStudent):
    torch.manual_seed(69)

    double_teacher = common_sense_overlapped_double_teacher(
        in_size=INPUT_DIMENSION,
        out_size=OUTPUT_DIMENSION,
        hid_size=TEACHER_HIDDEN_UNITS,
        overlap=overlap,
    )

    test_set = double_teacher.sample_batch(
        TEST_SIZE, return_both_teachers=True, output_noise_std=LABELS_NOISE_STD
    )

    test_losses_pre_switch = train_student(
        student=student, double_teacher=double_teacher, test_set=test_set
    )

    student.flip_switch()
    double_teacher.flip_switch()

    test_losses_post_switch = train_student(
        student=student, double_teacher=double_teacher, test_set=test_set
    )

    return test_losses_pre_switch, test_losses_post_switch


def main():
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    plot_params = dict(linewidth=1.0, alpha=0.7)

    original_student = TwoHeadStudent(
        in_size=INPUT_DIMENSION,
        out_size=OUTPUT_DIMENSION,
        hid_size=STUDENT_HIDDEN_UNITS,
    )

    for overlap in (0.0, 0.2, 0.5, 0.8, 1.0):
        print(f"running experiment for {overlap=}")
        test_losses_pre_switch, test_losses_post_switch = continual_learning_experiment(
            overlap=overlap, student=copy.deepcopy(original_student)
        )
        test_losses = {}
        for task in ("first_task", "second_task"):
            test_losses[task] = (
                test_losses_pre_switch[task] + test_losses_post_switch[task]
            )

        # Fixing SGD iteration index
        pre_index = np.array(test_losses_pre_switch["index"])
        pre_index += pre_index[1]
        post_index = (
            np.array(test_losses_post_switch["index"]) + pre_index[-1] + pre_index[0]
        )
        SGD_iteration = list(pre_index) + list(post_index)

        for i, task in enumerate(test_losses.keys()):
            axes[i].plot(SGD_iteration, test_losses[task], label=overlap, **plot_params)
            axes[i].set_yscale("log")

    for i, task in enumerate(test_losses.keys()):
        axes[i].title.set_text(task)
        axes[i].axvline(x=post_index[0], alpha=0.5, color="black")

    plt.legend(title="Overlap")
    plt.xlabel("SGD iteration")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig("whole_training.png", dpi=140)


if __name__ == "__main__":
    main()
