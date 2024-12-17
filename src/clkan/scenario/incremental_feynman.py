from pathlib import Path

import h5py
import numpy as np
import torch
from claiutil.datasets import orthogonalize_tasks
from torch.utils.data import ConcatDataset, TensorDataset


def load_incremental_feynman(
    datasets_root: Path,
    filename: str,
    shuffle: bool,
    joint: bool,
    standardize: bool,
    min_max_normalize: bool,
):
    """Load the incremental Feynman dataset from the h5 file.

    * In features: 83
    * Out features: 34

    :param datasets_root: The directory containing ``kanlike_feynaman_no_units.h5``.
    :param shuffle: Should the order of the equations be shuffled, defaults to True
    :return: The train and test streams and the masks for the tasks
    """
    datasets_root = Path(datasets_root)
    eval_percentage = 0.1

    # Load the datasets from the h5 file
    train_stream = []
    test_stream = []
    with h5py.File(datasets_root / filename, "r") as f:
        for dataset in f.keys():
            x = torch.tensor(np.array(f[dataset]["x"])).float()
            y = torch.tensor(np.array(f[dataset]["y"])).view(-1, 1).float()

            if standardize:
                # Normalize x and y to have zero mean and unit variance
                x = (x - x.mean(dim=0)) / x.std(dim=0)
                y = (y - y.mean(dim=0)) / y.std(dim=0)

            if min_max_normalize:
                # Normalize x and y to have values between 0 and 1
                x = (x - x.min(dim=0).values) / (
                    x.max(dim=0).values - x.min(dim=0).values
                )
                y = (y - y.min(dim=0).values) / (
                    y.max(dim=0).values - y.min(dim=0).values
                )

            num_samples = x.shape[0]
            num_eval_samples = int(num_samples * eval_percentage)
            num_train_samples = num_samples - num_eval_samples

            train_stream.append((x[:num_train_samples], y[:num_train_samples]))
            test_stream.append((x[num_train_samples:], y[num_train_samples:]))

    train_stream, _, task_mask = orthogonalize_tasks(train_stream)
    test_stream, _, _ = orthogonalize_tasks(test_stream)

    # Shuffle the orders
    if shuffle:
        shuffle_indices = np.random.permutation(len(train_stream))
        train_stream = [train_stream[i] for i in shuffle_indices]
        test_stream = [test_stream[i] for i in shuffle_indices]
        task_mask = task_mask[shuffle_indices]

    train_stream = [TensorDataset(x, y) for x, y in train_stream]
    test_stream = [TensorDataset(x, y) for x, y in test_stream]

    if joint:
        train_stream = [ConcatDataset(train_stream)]
        test_stream = [ConcatDataset(test_stream)]
        task_mask = torch.ones((1, task_mask.shape[1]), dtype=torch.bool)

    return train_stream, test_stream, task_mask
