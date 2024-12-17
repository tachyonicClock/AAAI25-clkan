"""This module is used to provide utility functions for PyTorch."""

import math
from os import cpu_count
from typing import Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


class BatchedTensorDataset(TensorDataset):
    """A :class:`torch.utils.data.TensorDataset` that supports batched indexing.

    ..  note::

        When working with a :class:`DataLoader`, the :meth:`collate_fn` method
        should be used to ensure that the batch is correctly constructed:

    >>> dataset = BatchedTensorDataset(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
    >>> loader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=2)
    >>> for batch in loader:
    ...     print(batch)
    (tensor([1, 2]), tensor([4, 5]))
    (tensor([3]), tensor([6]))

    """

    def __getitems__(self, indices: Sequence[int]) -> Tuple[Tensor, ...]:
        """Get a batch of items from the dataset."""
        return tuple(tensor[indices] for tensor in self.tensors)

    def collate_fn(self, args: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Use this function as the ``collate_fn`` in the :class:`DataLoader`."""
        return args


@torch.no_grad()
def preload(
    dataset: Dataset[Tuple[Tensor | int | float | bool, ...]], num_workers: int = -1
) -> TensorDataset:
    """Preload a dataset into memory.

    >>> from torch.utils.data import Subset
    >>> dataset = TensorDataset(torch.tensor([1, 2, 3]))
    >>> preloaded = preload_dataset(Subset(dataset, [0, 1]), 100)
    >>> preloaded.tensors[0]
    tensor([1, 2])

    :param dataset: The dataset to preload.
    :param num_workers: The number of workers to use for loading the dataset.
    :return: The preloaded dataset.
    """
    if isinstance(dataset, BatchedTensorDataset):
        return dataset
    elif isinstance(dataset, TensorDataset):
        return BatchedTensorDataset(*dataset.tensors)

    # Collect the shapes and dtypes of the first instance
    dataset_len = len(dataset)  # type: ignore # Assume dataset is Sized
    first_tuple = dataset[0]
    instance_shapes = []
    instance_dtypes = []
    instance_size = 0
    for tensor in first_tuple:
        if not isinstance(tensor, Tensor):
            tensor = torch.tensor(tensor)
        instance_shapes.append(tensor.shape)
        instance_dtypes.append(tensor.dtype)
        instance_size += tensor.element_size() * tensor.nelement()

    # Allocate the entire dataset
    tensors = [
        torch.empty(dataset_len, *shape, dtype=dtype)
        for shape, dtype in zip(instance_shapes, instance_dtypes, strict=True)
    ]

    # Load the dataset into memory
    batch_size = math.ceil(dataset_len / 100)
    for batch_id, batch in enumerate(
        DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers if num_workers >= 0 else (cpu_count() or 0) // 2,
        )
    ):
        start = batch_id * batch_size
        for i, tensor in enumerate(batch):
            tensors[i][start : start + len(tensor)] = (  # type: ignore
                tensor if isinstance(tensor, Tensor) else torch.tensor(tensor)
            )

    return BatchedTensorDataset(*tensors)
