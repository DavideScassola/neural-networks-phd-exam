import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from src.tooling.architectures import *
from src.tooling.data.dataset import SupervisedLearingDataset

N = 10_000
INPUT_DIMENSION = 500
TEACHER_HIDDEN_UNITS = 1
STUDENT_HIDDEN_UNITS = 2
OUTPUT_DIMENSION = 1
TRAIN_PROPORTION = 0.8
FIRST_HEAD_EPOCHS = 100
SECOND_HEAD_EPOCHS = 100
TRAIN_LOADER_PARAMS = dict(batch_size=32, shuffle=True)


def train_epoch(
    *,
    student: TwoHeadStudent,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    loss_fn,
    head: int,
):
    student.train()
    # Metrics
    running_train_loss = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out = student(X_batch, return_both_heads=True)[head]
        loss = loss_fn(out, y_batch.view((1, 1)).detach())
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item())
    return sum(running_train_loss) / len(train_loader.dataset)


def evaluate_on_test(
    *, student: TwoHeadStudent, dataset: SupervisedLearingDataset, loss_fn, head: int
):
    with torch.no_grad():
        student.train(False)
        return loss_fn(student(dataset.X_test)[head], dataset.y_test).mean().item()


def train_first_head(*, student: TwoHeadStudent, dataset: SupervisedLearingDataset):
    optimizer = torch.optim.SGD(student.trainable_parameters(lr=1), lr=1)
    loss_fn = nn.MSELoss()

    train_loader = dataset.get_train_loader(TRAIN_LOADER_PARAMS)

    epoch_train_losses = []
    epoch_test_losses = []

    for _ in range(FIRST_HEAD_EPOCHS):
        train_loss = train_epoch(
            student=student,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=loss_fn,
            head=0,
        )
        test_loss = evaluate_on_test(
            student=student, dataset=dataset, loss_fn=loss_fn, head=0
        )

        epoch_train_losses.append(train_loss)
        epoch_test_losses.append(test_loss)

        print(f"{train_loss=}")

    return epoch_train_losses, epoch_test_losses


def train_second_head(
    *,
    student: TwoHeadStudent,
    dataset_teacher1: SupervisedLearingDataset,
    dataset_teacher2: SupervisedLearingDataset,
):
    optimizer = torch.optim.SGD(
        student.trainable_parameters(lr=1), lr=1
    )  # TODO: change lr according to appendix
    loss_fn = nn.MSELoss()

    train_loader = dataset_teacher2.get_train_loader(TRAIN_LOADER_PARAMS)

    epoch_train_losses = []
    epoch_test_losses_dataset1 = []
    epoch_test_losses_dataset2 = []

    student.flip_switch()
    assert False

    for _ in range(SECOND_HEAD_EPOCHS):
        train_loss = train_epoch(
            student=student,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=loss_fn,
            head=0,
        )
        dataset1_test_loss = evaluate_on_test(
            student=student, dataset=dataset_teacher1, loss_fn=loss_fn, head=0
        )
        dataset2_test_loss = evaluate_on_test(
            student=student, dataset=dataset_teacher2, loss_fn=loss_fn, head=1
        )

        epoch_train_losses.append(train_loss)
        epoch_test_losses_dataset1.append(dataset1_test_loss)
        epoch_test_losses_dataset2.append(dataset2_test_loss)

        print(f"{train_loss=}")

    return epoch_train_losses, epoch_test_losses_dataset1, epoch_test_losses_dataset2


def get_teacher_dataset(*, double_teacher: DoubleTeacher, head: int):
    X1 = torch.normal(0.0, 1.0, size=(N, INPUT_DIMENSION))
    y1 = double_teacher(X1)[head]
    return SupervisedLearingDataset(X=X1, y=y1, train_proportion=TRAIN_PROPORTION)


def contiual_learning_experiment():
    # TODO: parametrize overlap
    double_teacher = DoubleTeacher(
        in_size=INPUT_DIMENSION,
        out_size=OUTPUT_DIMENSION,
        hid_size=TEACHER_HIDDEN_UNITS,
    )

    dataset_teacher1 = get_teacher_dataset(double_teacher=double_teacher, head=0)
    dataset_teacher2 = get_teacher_dataset(double_teacher=double_teacher, head=1)

    student = TwoHeadStudent(
        in_size=INPUT_DIMENSION,
        out_size=OUTPUT_DIMENSION,
        hid_size=STUDENT_HIDDEN_UNITS,
    )

    train_first_head(student=student, dataset=dataset_teacher1)

    train_second_head(
        student=student,
        dataset_teacher1=dataset_teacher1,
        dataset_teacher2=dataset_teacher2,
    )


def main():
    contiual_learning_experiment()
