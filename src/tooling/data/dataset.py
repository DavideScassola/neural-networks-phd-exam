#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class SupervisedLearingDataset:

    def __init__(
        self, *, x: torch.Tensor, y: torch.Tensor, train_proportion: float
    ) -> None:
        """ Generate a Supervised Learning Dataset with training and testing input->labels """

        if len(x) != len(y):
            raise ValueError("inputs x and labels y should be of the same length")
        self._x = x
        self._y = y
        self.train_proportion = train_proportion
        train_size = int(len(x) * train_proportion)
        self.x_train = x[:train_size]
        self.y_train = y[:train_size]
        self.x_test = x[train_size:]
        self.y_test = y[train_size:]


    def get_train_loader(
        self, **loader_kwargs
    ) -> DataLoader:
        """ Return a DataLoader object containing the Supervised Learning Dataset """

        return DataLoader(TensorDataset(self.x_train, self.x_test), **loader_kwargs)