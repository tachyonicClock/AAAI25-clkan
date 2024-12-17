"""Utilities for continual learning when using PyTorch datasets."""

from functools import cache, partial
from typing import Any, Callable, Iterable, Sequence, Set, Sized, Tuple, TypeVar

import torch
from torch import BoolTensor, LongTensor, Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset

_ClfDataset = Dataset[Tuple[Tensor, Tensor]]
_ClassSchedule = Sequence[Set[int]]
_TaskStream = Sequence[_ClfDataset]
"""A sequence of sets containing class indices defining task order and composition."""


_FlattenT = TypeVar("_FlattenT")


def _flatten(list_of_lists: Iterable[Iterable[_FlattenT]]) -> Sequence[_FlattenT]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def _is_unique_consecutive_from_zero(tensor: Tensor) -> bool:
    """Check if a tensor contains unique consecutive integers starting from 0.

    >>> _is_unique_consecutive_from_zero(torch.tensor([0, 1, 2]))
    True
    >>> _is_unique_consecutive_from_zero(torch.tensor([0, 1, 3]))
    False
    >>> _is_unique_consecutive_from_zero(torch.tensor([0, 1, 1]))
    False

    :param tensor: The tensor to check.
    :return: True if the tensor contains unique consecutive integers starting from 0,
    """
    unique_values: Tensor = tensor.unique()  # type: ignore
    return tensor.numel() == unique_values.numel() and unique_values.equal(
        torch.arange(unique_values.numel())
    )


def get_targets(dataset: _ClfDataset) -> LongTensor:
    """Return the targets of a dataset as a 1D tensor.

    * If the dataset has a `targets` attribute, it is used.
    * Otherwise, the targets are extracted from the dataset by iterating over it.

    :param dataset: The dataset to get the targets from.
    :return: A 1D tensor containing the targets of the dataset.
    """
    assert isinstance(dataset, Sized)

    # If possible use the dataset's targets
    if hasattr(dataset, "targets") and isinstance(dataset.targets, torch.Tensor):
        assert dataset.targets.dim() == 1, "Targets should be a 1D tensor"
        return LongTensor(dataset.targets)
    elif hasattr(dataset, "targets") and isinstance(dataset.targets, list):
        return LongTensor(dataset.targets)

    # Otherwise loop over the dataset to get the labels
    labels = LongTensor(len(dataset))
    for i in range(len(dataset)):
        labels[i] = int(dataset[i][1])
    return labels


def are_tasks_disjoint(class_schedule: _ClassSchedule) -> bool:
    """Check if the classes in each task are disjoint.

    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :return: True if the classes in each task are disjoint, False otherwise.
    """
    classes = _flatten(class_schedule)
    unique_classes = set(classes)
    return len(classes) == len(unique_classes)


def are_tasks_equally_sized(class_schedule: _ClassSchedule) -> bool:
    """Check if number of classes in each task are equal.

    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :return: True if the classes in each task are equally sized, False otherwise.
    """
    if len(class_schedule) == 0:
        return True
    task_size = len(class_schedule[0])
    return all(len(task) == task_size for task in class_schedule)


def partition_by_schedule(
    dataset: _ClfDataset, class_schedule: _ClassSchedule
) -> _TaskStream:
    """Divide a dataset into multiple datasets based on a class schedule.

    In class incremental learning, a task is a dataset containing a subset of
    the classes in the original dataset. This function divides a dataset into
    multiple tasks, each containing a subset of the classes.

    :param dataset: The dataset to divide.
    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :return: A list of datasets, each corresponding to a task.
    """
    targets = get_targets(dataset)
    task_datasets = []
    for classes in class_schedule:
        mask = targets.unsqueeze(1).eq(LongTensor(list(classes))).any(dim=1)
        indices = torch.nonzero(mask).squeeze(1)
        subset = Subset(dataset, indices.tolist())
        subset.targets = targets[indices]  # type: ignore
        assert isinstance(subset, Sized), "Subset should be a Sized object"
        task_datasets.append(subset)

    return task_datasets


def class_incremental_split(
    dataset: _ClfDataset,
    num_tasks: int,
    shuffle_tasks: bool = True,
    generator: torch.Generator = torch.default_generator,
) -> Tuple[_TaskStream, _ClassSchedule]:
    """Divide a dataset into multiple tasks for class incremental learning.

    >>> from torch.utils.data import TensorDataset
    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = torch.tensor([0, 1, 2, 3])
    >>> dataset = TensorDataset(x, y)
    >>> tasks, schedule = class_incremental_split(dataset, 2, shuffle_tasks=False)
    >>> schedule
    [{0, 1}, {2, 3}]
    >>> tasks[0][0]
    (tensor([1, 2]), tensor(0))
    >>> tasks[1][0]
    (tensor([5, 6]), tensor(2))


    :param dataset: The dataset to divide.
    :param num_tasks: The number of tasks to divide the dataset into.
    :param shuffle_tasks: When False, the classes occur in numerical order of their
        labels. When True, the classes are shuffled.
    :param generator: The random number generator used for shuffling, defaults
        to torch.default_generator
    :return: A tuple containing the list of tasks and the class schedule.
    """
    targets = get_targets(dataset)
    unique_labels: Tensor = targets.unique()  # type: ignore
    if not _is_unique_consecutive_from_zero(unique_labels):
        raise ValueError("Labels should be consecutive integers starting from 0")
    num_classes = unique_labels.numel()
    class_schedule = class_incremental_schedule(
        num_classes, num_tasks, shuffle_tasks, generator
    )
    return partition_by_schedule(dataset, class_schedule), class_schedule


def class_incremental_schedule(
    num_classes: int,
    num_tasks: int,
    shuffled: bool = True,
    generator: torch.Generator = torch.default_generator,
) -> _ClassSchedule:
    """Returns a class schedule for class incremental learning.

    >>> class_incremental_schedule(9, 3, shuffled=False)
    [{0, 1, 2}, {3, 4, 5}, {8, 6, 7}]

    >>> class_incremental_schedule(9, 3, generator=torch.Generator().manual_seed(0))
    [{8, 0, 2}, {1, 3, 7}, {4, 5, 6}]

    :param num_classes: The number of classes in the dataset.
    :param num_tasks: The number of tasks to divide the classes into.
    :param shuffled: When False, the classes occur in numerical order of their
        labels. When True, the classes are shuffled.
    :param generator: The random number generator used for shuffling, defaults
        to torch.default_generator
    :return: A list of lists of classes for each task.
    """
    if num_classes < num_tasks:
        raise ValueError("Cannot split classes into more tasks than classes")
    if num_classes == 0 or num_tasks == 0 or num_classes % num_tasks != 0:
        raise ValueError("Number of classes should be divisible by the number of tasks")

    classes = torch.arange(num_classes)
    if shuffled:
        classes = classes[torch.randperm(num_classes, generator=generator)]

    task_size = num_classes // num_tasks
    return [
        set(classes[i : i + task_size].tolist())
        for i in range(0, num_classes, task_size)
    ]


def class_schedule_to_task_mask(
    class_schedule: _ClassSchedule, num_classes: int
) -> BoolTensor:
    """Convert a class schedule to a list of boolean masks.

    This is useful when implementing multi-headed neural networks for task
    incremental learning.

    >>> class_schedule_to_task_mask([{0, 1}, {2, 3}], 4)
    tensor([[ True,  True, False, False],
            [False, False,  True,  True]])

    :param num_classes: The total number of classes.
    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :return: A boolean mask of shape (num_tasks, num_classes)
    """
    min_class = min(map(min, class_schedule), default=-1)
    max_class = max(map(max, class_schedule), default=-1)
    if not (0 <= min_class < num_classes) or not (0 <= max_class < num_classes):
        raise ValueError(
            "Classes in the schedule should be within the range of num_classes"
        )

    task_mask = torch.zeros(len(class_schedule), num_classes, dtype=torch.bool)
    for i, classes in enumerate(class_schedule):
        task_mask[i, list(classes)] = True
    return BoolTensor(task_mask)


def infer_class_schedule(stream: Sequence[_ClfDataset]) -> _ClassSchedule:
    """Infer the class schedule from a stream of datasets.

    This can be slow for large datasets if the only way to get the targets
    is by iterating over the entire dataset. It is preferable to keep track of
    the class schedule and reuse it when needed.

    :param stream: A list of datasets.
    :return: A class schedule for the datasets.
    """
    return [set(get_targets(dataset).tolist()) for dataset in stream]


class NoAvalancheError(RuntimeError):
    """Error raised when avalanche cannot be imported."""

    def __init__(self, func_name: str):
        """Create a new NoAvalancheError.

        :param func_name: The name of the function that requires Avalanche.
        """
        super().__init__(
            f"The continual learning library Avalanche is needed for `{func_name}`."
            "You may install it by following the instructions at "
            "https://avalanche.continualai.org/"
        )


def to_avalanche_nc_scenario(
    class_schedule: _ClassSchedule,
    train_stream: Sequence[_ClfDataset],
    test_stream: Sequence[_ClfDataset],
) -> Any:
    """Converts a class incremental schedule to an Avalanche benchmark scenario.

    :param class_schedule: A sequence of sets containing class indices defining
        task order and composition.
    :param train_stream: A list of training datasets.
    :param test_stream: A list of testing datasets.
    :raises ValueError: If the number of tasks does not match the number of datasets.
    :raises ImportError: If Avalanche is not installed.
    :return: An Avalanche benchmark scenario.
    """
    num_tasks = len(class_schedule)
    fixed_class_order = _flatten(class_schedule)
    if len(fixed_class_order) == 0:
        raise ValueError("Needs non-empty class schedule.")
    if not are_tasks_disjoint(class_schedule):
        raise ValueError("Needs disjoint tasks in class schedule.")
    if not are_tasks_equally_sized(class_schedule):
        raise ValueError("Needs equally sized tasks in class schedule.")
    if num_tasks != len(train_stream) or num_tasks != len(test_stream):
        raise ValueError("Needs lengths of inputs to match the number of tasks.")
    try:
        from avalanche.benchmarks import nc_benchmark  # type: ignore

        return nc_benchmark(
            ConcatDataset(train_stream),
            ConcatDataset(test_stream),
            n_experiences=num_tasks,
            fixed_class_order=fixed_class_order,
            task_labels=True,
            shuffle=False,
        )
    except ImportError as err:
        raise NoAvalancheError(to_avalanche_nc_scenario.__name__) from err


@torch.no_grad()
def orthogonalize_tasks(
    stream: Sequence[Tuple[Tensor, Tensor]],
    orthogonalize_inputs: bool = True,
    orthogonalize_outputs: bool = True,
) -> Tuple[Sequence[Tuple[Tensor, Tensor]], BoolTensor, BoolTensor]:
    """Make the tasks in a stream of datasets orthogonal.

    Meaning each task's inputs and outputs do not overlap with other tasks. This`
    is achieved by expanding the input/output space to contain a separate entry
    for each task. The additional entries are zeroed out for tasks they do not
    belong to.

    Example orthogonalizing inputs and outputs:
    >>> task_0 = (torch.tensor([[0], [1]]), torch.tensor([[0], [0]]))
    >>> task_1 = (torch.tensor([[2], [3]]), torch.tensor([[1], [1]]))
    >>> stream, in_mask, out_mask = orthogonalize_tasks([task_0, task_1])
    >>> stream
    [(tensor([[0, 0], [1, 0]]), tensor([[0, 0], [0, 0]])),
     (tensor([[0, 2], [0, 3]]), tensor([[0, 1], [0, 1]]))]
    >>> in_mask
    tensor([[ True, False],
            [False,  True]])
    >>> out_mask
    tensor([[ True, False],
            [False,  True]])

    Example orthogonalizing only inputs:
    >>> stream, in_mask, out_mask = orthogonalize_tasks([task_0, task_1],
    ...                                           orthogonalize_inputs=False)
    >>> stream
    [(tensor([[0], [1]]), tensor([[0, 0], [0, 0]])),
     (tensor([[2], [3]]), tensor([[0, 1], [0, 1]]))]
    >>> in_mask
    tensor([[True],
            [True]])
    >>> out_mask
    tensor([[ True, False],
            [False,  True]])

    Example orthogonalizing only outputs:
    >>> stream, in_mask, out_mask = orthogonalize_tasks([task_0, task_1],
    ...                                                 orthogonalize_outputs=False)
    >>> stream
    [(tensor([[0, 0], [1, 0]]), tensor([[0], [0]])),
     (tensor([[0, 2], [0, 3]]), tensor([[1], [1]]))]
    >>> in_mask
    tensor([[ True, False],
            [False,  True]])
    >>> out_mask
    tensor([[True],
            [True]])

    Example not orthogonalizing:
    >>> stream, in_mask, out_mask = orthogonalize_tasks([task_0, task_1],
    ...                                                 orthogonalize_inputs=False,
    ...                                                 orthogonalize_outputs=False)
    >>> stream
    [(tensor([[0], [1]]), tensor([[0], [0]])),
        (tensor([[2], [3]]), tensor([[1], [1]]))]
    >>> in_mask
    tensor([[True],
            [True]])
    >>> out_mask
    tensor([[True],
            [True]])

    When not orthogonalizing, the input and output dimensions of each task should be
    the same. If they are not, a ValueError is raised.

    :param stream: A sequence of datasets represented as tuples of inputs and
        outputs. The input tensor should have shape ``(batch_size, input_dim)``
        and the target tensor should have shape ``(batch_size, target_dim)``.
    :orthogonalize_inputs: If False, the input tensors are not expanded.
    :orthogonalize_outputs: If False, the target tensors are not expanded.
    :return: A tuple containing the orthogonal stream and the task mask.

        - ``stream``: A sequence of tensors that have been expanded.

          - The input tensor has shape ``(batch_size, new_input_dim)``. Where
            if ``orthogonalize_inputs`` is True, the ``new_input_dim`` is the sum
            of the ``input_dim`` of each task. Otherwise, it is the same as the
            original ``input_dim``.

          - The target tensor has shape `(batch_size, new_target_dim)`. Where if
            ``orthogonalize_outputs`` is True, the ``new_target_dim`` is the sum of
            the ``target_dim`` of each task. Otherwise, it is the same as the
            original ``target_dim``.

        -   ``in_mask``: A boolean tensor of shape `(num_tasks, new_input_dim)`
            where each row is a mask for the input dimensions of a task.

        -   ``out_mask``: A boolean tensor of shape `(num_tasks, new_target_dim)`
            where each row is a mask for the target dimensions of a task.
    """
    if not all(x_set.ndim == 2 for x_set, _ in stream):
        raise ValueError("Each task's input should have shape (batch_size, input_dim)")
    if not all(y_set.ndim == 2 for _, y_set in stream):
        raise ValueError(
            "Each task's target should have shape (batch_size, target_dim)"
        )
    if not all(x_set.shape[0] == y_set.shape[0] for x_set, y_set in stream):
        raise ValueError("Each task's input and target should have the same batch size")

    def _orthogonalize_partial_stream(
        partial_stream: Sequence[Tensor],
    ) -> Tuple[Sequence[Tensor], Tensor]:
        new_dim = sum(x.shape[1] for x in partial_stream)
        out_partial_stream = []
        task_mask = torch.zeros((len(partial_stream), new_dim), dtype=torch.bool)

        offset = 0
        for task_id, task in enumerate(partial_stream):
            # Create a new tensor with the expanded dimension and set the mask
            new_x = torch.zeros((task.shape[0], new_dim), dtype=task.dtype)
            new_x[:, offset : offset + task.shape[1]] = task
            task_mask[task_id, offset : offset + task.shape[1]] = True

            offset += task.shape[1]
            out_partial_stream.append(new_x)

        return out_partial_stream, task_mask

    x_stream: Sequence[Tensor] = [x for x, _ in stream]
    y_stream: Sequence[Tensor] = [y for _, y in stream]
    n_tasks: int = len(x_stream)
    x_dims = {x.shape[1] for x in x_stream}
    y_dims = {y.shape[1] for y in y_stream}

    if orthogonalize_inputs:
        x_stream, x_mask = _orthogonalize_partial_stream(x_stream)
    elif len(x_dims) == 1:
        (x_dim,) = x_dims
        x_mask = torch.ones((n_tasks, x_dim), dtype=torch.bool)
    else:
        raise ValueError(
            "Input dimensions should be the same when not orthogonalizing. "
            f"Got dimensions {[x.shape[1] for x in x_stream]}"
        )

    if orthogonalize_outputs:
        y_stream, y_mask = _orthogonalize_partial_stream(y_stream)
    elif len(y_dims) == 1:
        (y_dim,) = y_dims
        y_mask = torch.ones((n_tasks, y_dim), dtype=torch.bool)
    else:
        raise ValueError(
            "Target dimensions should be the same when not orthogonalizing. "
            f"Got dimensions {[y.shape[1] for y in y_stream]}"
        )

    return (
        list(zip(x_stream, y_stream, strict=True)),
        BoolTensor(x_mask),
        BoolTensor(y_mask),
    )


def permute_tensor(x: Tensor, seed: int) -> Tensor:
    """Permute the elements of a tensor along the first dimension.

    The permutation is deterministic and is based on the given seed.

    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    >>> permute_tensor(x, seed=0)
    tensor([[3, 6],
            [4, 1],
            [2, 5]])

    :param x: The tensor to permute.
    :param seed: The seed used for the permutation.
    :return: The permuted tensor.
    """

    @cache
    def _random_permutation(n: int, seed: int) -> Tensor:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randperm(n, generator=generator)

    return x.view(-1)[_random_permutation(x.numel(), seed)].view(x.size())


class TransformDecoratorDataset(_ClfDataset):
    """Wrap a dataset with a transformation function."""

    def __init__(
        self,
        dataset: _ClfDataset,
        transform: Callable[[Tensor], Tensor],
        targets: Tensor,
    ) -> None:
        """Create a new ``TransformDecoratorDataset``.

        :param dataset: The dataset to wrap. Must be a dataset of tuples ``(x, ...)``
            where ``x`` is the input tensor and the rest of the tuple is ignored.
        :param transform: The transformation function to apply to the dataset.
        :param targets: The targets of the dataset.
        """
        self.dataset = dataset
        self.transform = transform
        self.targets = targets

    def __len__(self) -> int:
        """The length of the underlying dataset."""
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get the transformed item at the given index."""
        x = self.dataset[index][0]
        return self.transform(x), self.targets[index]


def permuted_stream(
    dataset: _ClfDataset, num_tasks: int, seed: int
) -> Tuple[Sequence[_ClfDataset], Sequence[Set[int]]]:
    """Create a stream of datasets with permuted inputs.

    >>> from torch.utils.data import TensorDataset
    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    >>> y = torch.tensor([0, 1, 2])
    >>> dataset = TensorDataset(x, y)
    >>> stream, schedule = permuted_stream(dataset, 3, seed=0)
    >>> len(stream)
    3
    >>> schedule
    [{0, 1, 2}, {3, 4, 5}, {8, 6, 7}]
    >>> stream[0][0]
    (tensor([1, 2]), tensor(0))
    >>> stream[1][0]
    (tensor([2, 1]), tensor(3))
    >>> stream[2][0]
    (tensor([1, 2]), tensor(6))


    :param dataset: The dataset to permute. Must be a dataset of tuples ``(x, ...)``
    :param num_tasks: The number of tasks to create.
    :param seed: The seed used for the permutation.
    :return: A tuple containing the permuted stream and the class schedule.
    """
    targets = get_targets(dataset)
    unique_labels: Tensor = targets.unique()  # type: ignore
    n_classes = unique_labels.numel()

    # Ensure labels are consecutive integers starting from 0
    if not unique_labels.equal(torch.arange(n_classes)):
        raise ValueError("Labels should be consecutive integers starting from 0")

    stream = []
    schedule = []
    for task_id in range(num_tasks):
        task_targets = n_classes * task_id + targets
        schedule.append(set(range(n_classes * task_id, n_classes * (task_id + 1))))
        permutation = partial(permute_tensor, seed=seed + task_id)
        stream.append(TransformDecoratorDataset(dataset, permutation, task_targets))

    return stream, schedule
