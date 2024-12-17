"""This module contains domain incremental scenarios for the regression task."""

from pathlib import Path
from typing import Sequence, Tuple

import h5py
import torch
from claiutil.datasets import orthogonalize_tasks
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset


class HDF5Dataset(Dataset):
    def __init__(
        self,
        path: str,
        x_dataset_key: str = "x_features",
        y_dataset_key: str = "y_targets",
    ):
        self.file = h5py.File(path, "r")
        self.x_data = self.file[x_dataset_key]
        self.y_data = self.file[y_dataset_key]
        if len(self.x_data) != len(self.y_data):
            raise ValueError("The x and y datasets must have the same length.")

    def __getitem__(self, index):
        y = self.y_data[index]
        x = self.x_data[index]
        return torch.tensor(x).float(), torch.tensor(y).float()

    def __len__(self):
        return len(self.x_data)


def from_hdf5(
    path: str, x_dataset_key: str = "x_features", y_dataset_key: str = "y_targets"
) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as file:
        x_data = file[x_dataset_key][()]
        y_data = file[y_dataset_key][()]
        return torch.tensor(x_data).float(), torch.tensor(y_data).float()


def task_incremental_regression(
    datasets_root: Path,
    hdf5_filename: str,
    num_tasks: int,
    test_proportion: float,
    orthogonalize_inputs: bool = False,
    orthogonalize_targets: bool = False,
):
    """Create a domain incremental regression scenario from a regression dataset.

    This scenario is constructed by splitting the dataset into tasks based on
    the index of the samples.

    :param datasets_root: Base path to look for the dataset.
    :param hdf5_filename: The name of the HDF5 file containing the dataset.
    :param num_tasks: The number of tasks to split the dataset into.
    :param test_proportion: The proportion of the training data to use as test data.
    :return: A tuple containing the train and test streams.
    """
    datasets_root = Path(datasets_root)
    x, y = from_hdf5(datasets_root / hdf5_filename)
    generator = torch.Generator().manual_seed(42)
    # Split the dataset into tasks
    num_samples = len(x)
    task_size = num_samples // num_tasks
    test_task_size = int(test_proportion * task_size)

    # Build the train stream by splitting the dataset into tasks
    train_tasks: Sequence[Tuple[Tensor, Tensor]] = []
    test_tasks: Sequence[Tuple[Tensor, Tensor]] = []
    for i in range(num_tasks):
        start = i * task_size
        end = start + task_size
        task_indices = torch.arange(start, end)
        task_indices = task_indices[
            torch.randperm(len(task_indices), generator=generator)
        ]
        train_indices = task_indices[test_task_size:]
        test_indices = task_indices[:test_task_size]
        train_tasks.append((x[train_indices], y[train_indices]))
        test_tasks.append((x[test_indices], y[test_indices]))

    train_stream, _, out_mask = orthogonalize_tasks(
        train_tasks, orthogonalize_inputs, orthogonalize_targets
    )
    test_stream, _, _ = orthogonalize_tasks(
        test_tasks, orthogonalize_inputs, orthogonalize_targets
    )

    train_stream = [TensorDataset(x, y) for x, y in train_stream]
    test_stream = [TensorDataset(x, y) for x, y in test_stream]

    return train_stream, test_stream, out_mask
