from typing import Any

import torch
from torch.utils.data import DataLoader


class SupervisedLearingDataset:
    def __init__(
        self, *, X: torch.Tensor, y: torch.Tensor, train_proportion: float
    ) -> None:
        if len(X) != len(y):
            raise ValueError("inputs X and labels y should be of the same length")
        self._X = X
        self._y = y
        self.train_proportion = train_proportion
        train_size = int(len(X) * train_proportion)
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_test = X[train_size:]
        self.y_test = y[train_size:]

    def get_train_loader(self, **loader_kwargs) -> DataLoader:
        return DataLoader(list(zip(self.X_train, self.X_test)), **loader_kwargs)
