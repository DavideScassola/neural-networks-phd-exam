#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

from src.tooling.architectures import DoubleTeacher
from src.tooling.architectures import TwoHeadStudent
from src.tooling.architectures import tensor_pair_from_overlap
import matplotlib.pyplot as plt
import tqdm
import random
import copy

N = 1_000_000
INPUT_DIMENSION = 500
TEACHER_HIDDEN_UNITS = 1
STUDENT_HIDDEN_UNITS = 2
OUTPUT_DIMENSION = 1
TRAIN_PROPORTION = 0.8
BATCH_SIZE = 1000
TEST_SIZE = 1000
LABELS_NOISE_STD = 0.01
FIRST_HEAD_BATCHES = N // BATCH_SIZE
SECOND_HEAD_BATCHES = FIRST_HEAD_BATCHES


def train(
    *,
    student: TwoHeadStudent,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    batches: int,
    teacher: DoubleTeacher,
):
    """Training for the student network."""

    student.train()

    # Metrics
    losses = {"first_head": [], "second_head": []}

    for _ in tqdm.tqdm(range(batches)):
        x_batch, y_batch = teacher.sample_batch(
            n=BATCH_SIZE, output_noise_std=LABELS_NOISE_STD
        )
        optimizer.zero_grad()
        out1, out2 = student(x_batch, return_both_heads=True)
        loss1, loss2 = loss_fn(out1, y_batch), loss_fn(out2, y_batch)
        loss = loss1 if not student._switch else loss2
        loss.backward()
        optimizer.step()
        losses["first_head"].append(loss1.item())
        losses["second_head"].append(loss2.item())

    return losses


def evaluate_on_test(
    *,
    student: TwoHeadStudent,
    double_teacher: DoubleTeacher,
    return_both_heads: bool,
):
    """Evaluate the student network on the teacher labels."""

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        student.train(False)

        if not return_both_heads:
            # Only when testing first head
            x_test, y_test = double_teacher.sample_batch(
                TEST_SIZE, return_both_teachers=False, output_noise_std=LABELS_NOISE_STD
            )
            return loss_fn(student(x_test, return_both_heads=False), y_test).item()
        else:
            x_test, y_test1, y_test2 = double_teacher.sample_batch(
                TEST_SIZE, return_both_teachers=True, output_noise_std=LABELS_NOISE_STD
            )
            out1, out2 = student(x_test, return_both_heads=True)
            loss1 = loss_fn(out1, y_test1).item()
            loss2 = loss_fn(out2, y_test2).item()
            return loss1, loss2


def train_student(
    *,
    student: TwoHeadStudent,
    double_teacher: DoubleTeacher,
):
    optimizer = torch.optim.SGD(student.trainable_parameters(lr=1), lr=1)
    loss_fn = nn.MSELoss()

    train_losses = train(
        student=student,
        optimizer=optimizer,
        teacher=double_teacher,
        loss_fn=loss_fn,
        batches=FIRST_HEAD_BATCHES,
    )

    return train_losses


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
    if overlap > 0.0:
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


def contiual_learning_experiment(*, overlap=0.0):

    double_teacher = overlapped_double_teacher(
        in_size=INPUT_DIMENSION,
        out_size=OUTPUT_DIMENSION,
        hid_size=TEACHER_HIDDEN_UNITS,
        overlap=overlap,
    )

    student = TwoHeadStudent(
        in_size=INPUT_DIMENSION,
        out_size=OUTPUT_DIMENSION,
        hid_size=STUDENT_HIDDEN_UNITS,
    )

    training_losses_pre_switch = train_student(
        student=student, double_teacher=double_teacher
    )

    test_loss1_pre_switch = evaluate_on_test(
        student=student, double_teacher=double_teacher, return_both_heads=False
    )

    student.flip_switch()
    double_teacher.flip_switch()

    training_losses_post_switch = train_student(
        student=student, double_teacher=double_teacher
    )

    test_loss1_post_switch, test_loss2_post_switch = evaluate_on_test(
        student=student, double_teacher=double_teacher, return_both_heads=True
    )

    print(f"{test_loss1_pre_switch=}")
    print(f"{test_loss1_post_switch=}")
    print(f"{test_loss2_post_switch=}")

    return (
        training_losses_pre_switch["first_head"]
        + training_losses_post_switch["first_head"],
        training_losses_pre_switch["second_head"]
        + training_losses_post_switch["second_head"],
    )


def deprecated_main():
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))

    for overlap in (0.0, 0.2, 0.5, 0.8, 1.0):    
        torch.manual_seed(69)

        print(f"running experiment for {overlap=}")
        first_head_losses, second_head_losses = contiual_learning_experiment(
            overlap=overlap
        )
        axes[0].plot(first_head_losses, linewidth=1.0, label=overlap, alpha=0.7)
        axes[1].plot(second_head_losses, linewidth=1.0, label=overlap, alpha=0.7)

    axes[0].title.set_text("first head")
    axes[1].title.set_text("second head")

    plt.legend(title="Overlap")
    plt.xlabel("SGD iteration")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig("whole_training.png", dpi=140)


def multiple_overlaps_test():
    og_student = TwoHeadStudent(
        in_size=INPUT_DIMENSION,
        out_size=OUTPUT_DIMENSION,
        hid_size=STUDENT_HIDDEN_UNITS,
    )

    for overlap in [0.0, 0.2, 0.5, 0.8, 1.0]:
        student = copy.deepcopy(og_student)

        torch.manual_seed(69)

        double_teacher = overlapped_double_teacher(
            in_size=INPUT_DIMENSION,
            out_size=OUTPUT_DIMENSION,
            hid_size=TEACHER_HIDDEN_UNITS,
            overlap=overlap,
        )

        training_losses_pre_switch = train_student(
            student=student, double_teacher=double_teacher
        )

        test_loss1_pre_switch = evaluate_on_test(
            student=student, double_teacher=double_teacher, return_both_heads=False
        )

        student.flip_switch()
        double_teacher.flip_switch()

        training_losses_post_switch = train_student(
            student=student, double_teacher=double_teacher
        )

        test_loss1_post_switch, test_loss2_post_switch = evaluate_on_test(
            student=student, double_teacher=double_teacher, return_both_heads=True
        )

        plt.cla()
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
        first_head_losses, second_head_losses = (
            training_losses_pre_switch["first_head"]
            + training_losses_post_switch["first_head"],
            training_losses_pre_switch["second_head"]
            + training_losses_post_switch["second_head"],
        )
        axes[0].plot(first_head_losses, linewidth=1.0)
        axes[0].set_yscale("log")
        axes[1].plot(second_head_losses, linewidth=1.0)
        axes[1].set_yscale("log")
        # plt.show()
        plt.suptitle(
            f"Training losses for student heads with {overlap} overlap in teacher weights",
            fontweight="bold",
            fontsize=20,
        )
        plt.savefig(f"whole_training_overlap_{overlap}.png", dpi=140)
        plt.cla()

    return test_loss1_pre_switch, test_loss1_post_switch, test_loss2_post_switch


def main():
    # sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))

    # for overlap in [0.0, 0.2, 0.5, 0.8, 1.0]:
    #     plt.cla()
    #     fig, axes = plt.subplots(1,2,figsize=(20, 10))
    #     first_head_losses, second_head_losses = contiual_learning_experiment(overlap=overlap)
    #     axes[0].plot(first_head_losses, linewidth=1.0)
    #     axes[0].set_yscale('log')
    #     axes[1].plot(second_head_losses, linewidth=1.0)
    #     axes[1].set_yscale('log')
    #     # plt.show()
    #     plt.suptitle(f"Training losses for student heads with {overlap} overlap in teacher weights")
    #     plt.savefig(f'whole_training_overlap_{overlap}.png', dpi=140)
    #     plt.cla()

    multiple_overlaps_test()


if __name__ == "__main__":
    main()
