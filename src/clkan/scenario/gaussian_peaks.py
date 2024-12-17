from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

import clkan.config as cfg


class TensorDataset(Dataset):
    def __init__(self, tensor: Tensor) -> None:
        assert tensor.dim() == 2
        self.tensor = tensor.float()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.tensor[index][0].view(-1), self.tensor[index][1].view(-1)

    def __len__(self) -> int:
        return self.tensor.size(0)


def _generate_gaussian_peaks(num_peaks: int, width: float, num: int) -> List[Tensor]:
    x_knots = torch.linspace(-1, 1, num_peaks + 1)
    datasets = []

    for x_start, x_end in zip(x_knots[:-1], x_knots[1:]):
        x_center = (x_start + x_end) / 2
        dataset = torch.zeros(num, 2)
        x = torch.rand(num) * (x_end - x_start) + x_start
        y = torch.exp(-0.5 * ((x - x_center) / (width / num_peaks)) ** 2)
        dataset[:, 0] = x
        dataset[:, 1] = y
        datasets.append(dataset)

    return datasets


def gaussian_peaks_dataset(
    config: cfg.GaussianPeaks,
) -> Tuple[List[Dataset], List[Dataset]]:
    """

    Example:
    -------
    >>> _ = torch.manual_seed(0)
    >>> config = cfg.GaussianPeaks(num_train_task_samples=5, num_test_task_samples=5)
    >>> benchmark = gaussian_peaks_dataset(config)
    >>> for experience in benchmark[0]:
    ...     print(f"{experience[0][0].item():0.2f}")
    -0.80
    -0.35
    -0.06
    0.41
    0.87
    """

    train_data = _generate_gaussian_peaks(
        config.peaks, config.width, config.num_train_task_samples
    )
    test_data = _generate_gaussian_peaks(
        config.peaks, config.width, config.num_test_task_samples
    )

    train_data = [TensorDataset(data) for data in train_data]
    test_data = [TensorDataset(data) for data in test_data]

    return train_data, test_data
