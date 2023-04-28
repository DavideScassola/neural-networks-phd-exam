#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional

import architectures as arc
import torch as th
from architectures import TwoHeadStudent
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

# from architectures import DoubleTeacher


def teacher_input_data(data_size: int = 1000, device: Optional[str] = None) -> Tensor:
    """Generate iid vectors to be fed to the teacher network."""

    return th.normal(0.0, 1.0, size=(data_size, 500), device=device)


def teacher_dataset(data_size: int, overlap: Tensor) -> DataLoader:
    """Generate labels to be used by the student during training or testing phases."""

    # Store the labels generated from both teachers
    first_labels = []
    second_labels = []

    # Initialize the teachers with a certain overlap in the first layer weights
    teachers = arc.goldt_school_from_overlap(out_size=1, overlap=overlap)

    # Get input iid gaussian distributed vectors
    input_data = teacher_input_data(data_size)

    for input in input_data:
        teacher_labels = teachers(input.view((1, input.size()[0])), True)
        first_labels += teacher_labels[0]
        second_labels += teacher_labels[1]

    first_labels = th.Tensor(first_labels)
    second_labels = th.Tensor(second_labels)

    return DataLoader(list(zip(input_data, first_labels, second_labels)))


def student_training_two_teachers(
    student: 'TwoHeadStudent', train_data: 'DataLoader',
):

    optimizer = th.optim.SGD(student.trainable_parameters(lr=1), lr=1)
    loss_fn = nn.MSELoss()

    # Metrics
    running_loss = []

    student.train()
    for index, data in enumerate(train_data):
        vec, first_label, second_label = data

        optimizer.zero_grad()

        if index == len(train_data.dataset)//2:
            student.flip_switch()

        out_1, out_2 = student(vec, return_both_heads=True)

        if index < len(train_data.dataset)//2:
            loss = loss_fn(out_1, first_label.view((1,1)).detach())
        else:
            loss = loss_fn(out_2, second_label.view((1,1)).detach())

        loss.backward()

        optimizer.step()

        running_loss.append(loss.item())

    avg_loss = sum(running_loss)/len(train_data.dataset)
        
    return running_loss, avg_loss


def student_training(
    student: TwoHeadStudent,
    train_data: DataLoader,
):
    optimizer = th.optim.SGD(student.trainable_parameters(lr=1), lr=1)
    loss_fn = nn.MSELoss()

    # Metrics
    running_loss = 0.0

    student.train()
    for data, label in train_data:
        optimizer.zero_grad()
        out = student(data, return_both_heads=False)
        loss = loss_fn(out, label.view((1,1)).detach())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_data.dataset)

    return avg_loss

if __name__ == '__main__':
    # teacher = arc.goldt_school(1)
    # student = arc.goldt_student(1)
    # teacher_data = teacher_input_data(5)
    # teacher_labels = teacher(teacher_data, False)
    # noise = th.randn(teacher_labels.size())
    # teacher_labels += noise
    # student_training_data = DataLoader(list(zip(teacher_data, teacher_labels)))

    dataset = teacher_dataset(100, th.tensor(0.))
    generalization_err, avg_loss = student_training_two_teachers(arc.goldt_student(1), dataset)