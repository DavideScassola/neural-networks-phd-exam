#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from src.tooling.architectures import DoubleTeacher
from src.tooling.architectures import TwoHeadStudent
from src.tooling.data.dataset import SupervisedLearingDataset
from src.tooling.architectures import tensor_pair_from_overlap
import matplotlib.pyplot as plt
import random

N = 1_000_000
INPUT_DIMENSION = 500
TEACHER_HIDDEN_UNITS = 1
STUDENT_HIDDEN_UNITS = 2
OUTPUT_DIMENSION = 1
TRAIN_PROPORTION = 0.8
BATCH_SIZE = 320
TEST_SIZE = 1000
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
    """Training epoch for the student network. Specify the head [0,1] to be trained."""

    student.train()

    # Metrics
    losses = []
    
    if student._switch:
        pass

    for _ in range(batches):
        x_batch, y_batch = teacher.sample_batch(n=BATCH_SIZE)
        optimizer.zero_grad()
        out = student(x_batch, return_both_heads=False)
        loss = loss_fn(out, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if random.random() > 0.95:
            print(f"{loss.item()=}")

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
                TEST_SIZE, return_both_teachers=False
            )
            return (
                loss_fn(student(x_test, return_both_heads=False), y_test).item()
            )
        else:
            x_test, y_test1, y_test2 = double_teacher.sample_batch(
                TEST_SIZE, return_both_teachers=True
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
        student=student, optimizer=optimizer, teacher=double_teacher, loss_fn=loss_fn, batches=FIRST_HEAD_BATCHES
    )

    return train_losses


def get_teacher_dataset(*, double_teacher: DoubleTeacher, teacher_index: int):
    """Generate iid vectors to be fed to the teacher network."""

    X1 = torch.normal(0.0, 1.0, size=(N, INPUT_DIMENSION))
    y1 = double_teacher(X1, return_both_teachers=True)[teacher_index]
    y1 += torch.randn(y1.size())
    return SupervisedLearingDataset(x=X1, y=y1, train_proportion=TRAIN_PROPORTION)


def overlapped_double_teacher(
    *, in_size: int, out_size: int, hid_size: int, overlap
) -> DoubleTeacher:
    return DoubleTeacher(
        in_size,
        hid_size,
        out_size,
        init_features_from=tensor_pair_from_overlap(overlap, in_size),
    )


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

    training_losses = train_student(
        student=student, double_teacher=double_teacher
    )

    test_loss1_pre_switch = evaluate_on_test(
        student=student, double_teacher=double_teacher, return_both_heads=False
    )
    
    plt.plot(training_losses, label = "pre switch")
    plt.legend()
    plt.savefig('pre_switch.png')
        
    student.flip_switch()
    double_teacher.flip_switch()

    training_losses = train_student(
        student=student, double_teacher=double_teacher
    )
    
    test_loss1_post_switch, test_loss2_post_switch = evaluate_on_test(
        student=student, double_teacher=double_teacher, return_both_heads=True
    )
    
    plt.cla()
    plt.plot(training_losses, label = "post switch")
    plt.legend()
    plt.savefig('post_switch.png')
    
    
    print(f"{test_loss1_pre_switch=}")
    print(f"{test_loss1_post_switch=}")
    print(f"{test_loss2_post_switch=}")

def main():
    contiual_learning_experiment(overlap=1.0)


if __name__ == "__main__":
    main()
