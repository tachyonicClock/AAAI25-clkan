"""Metrics for evaluating continual learning algorithms."""

import torch
from torch import Tensor


def _check_and_get_n_tasks(results: Tensor) -> int:
    if results.size(0) != results.size(1):
        raise ValueError("Results tensor must be square.")
    if results.ndim != 2:
        raise ValueError("Results tensor must be two-dimensional.")
    if results.size(0) < 2:
        raise ValueError("Results tensor must have at least two tasks.")
    return results.size(0)


def forward_transfer(results: Tensor) -> float:
    r"""Forward transfer measures the influence learning a task has on future tasks.

    Forward transfer metric as defined by Equation 4 of Díaz-Rodríguez et al. (2018).

    .. math::

        FWT=\frac{\sum_{i<j}^N R_{i, j}}{\frac{N(N-1)}{2}}

    >>> import torch
    >>> results = torch.tensor(
    ...     [[0.5, 0.6, 0.7],
    ...      [0.4, 0.5, 0.6],
    ...      [0.3, 0.4, 0.5]])
    >>> expected = (0.6 + 0.7 + 0.6) / 3
    >>> actual = forward_transfer(results)
    >>> print(f"{actual:.2f} == {expected:.2f}")
    0.63 == 0.63

    **References**:

    * Díaz-Rodríguez, N., Lomonaco, V., Filliat, D., & Maltoni, D. (2018).
      Don’t forget, there is more than forgetting: New metrics for Continual
      Learning (arXiv:1810.13166). arXiv. http://arxiv.org/abs/1810.13166

    :param results: A tensor of shape (``num_tasks``, ``num_tasks``) where
        ``results[i, j]`` is the performance of task ``j`` after training on task ``i``.
    :return: The forward transfer for the given task.
    """
    num_tasks = _check_and_get_n_tasks(results)
    return (
        torch.triu(results, diagonal=1).sum() / (num_tasks * (num_tasks - 1) / 2)
    ).item()


def backward_transfer(results: Tensor) -> float:
    r"""Backward transfer measures the influence learning a task has on previous tasks.

    Backward transfer metric as defined by Equation 3 of Díaz-Rodríguez et al. (2018).

    .. math::

       BWT=\frac{\sum_{i=2}^N \sum_{j=1}^{i-1}\left(R_{i, j}-R_{j, j}\right)}
       {\frac{N(N-1)}{2}}

    >>> import torch
    >>> results = torch.tensor(
    ...     [[0.5, 0.6, 0.7],
    ...      [0.4, 0.5, 0.6],
    ...      [0.3, 0.4, 0.5]])
    >>> expected = ((0.4 - 0.5) + (0.3 - 0.5) + (0.4 - 0.5)) / 3
    >>> actual = backward_transfer(results)
    >>> print(f"{actual:.2f} == {expected:.2f}")
    -0.13 == -0.13

    **References**:

    * Díaz-Rodríguez, N., Lomonaco, V., Filliat, D., & Maltoni, D. (2018).
      Don’t forget, there is more than forgetting: New metrics for Continual
      Learning (arXiv:1810.13166). arXiv. http://arxiv.org/abs/1810.13166

    :param results: A tensor of shape (``num_tasks``, ``num_tasks``) where
        ``results[i, j]`` is the performance of task ``j`` after training on
        task ``i``.
    :return: The backward transfer for the given task.
    """
    num_tasks = _check_and_get_n_tasks(results)

    total = 0.0
    for i in range(1, num_tasks):
        for j in range(0, i):
            total += (results[i, j] - results[j, j]).item()
    return total / (num_tasks * (num_tasks - 1) / 2)


def scenario_average(results: Tensor) -> float:
    r"""Average performance on past tasks over the course of training.

    This metric is usually used with accuracy and is defined by Equation 2 of
    Díaz-Rodríguez et al. (2018).

    .. math::

       A=\frac{\sum_{i \geq j}^N R_{i, j}}{\frac{N(N+1)}{2}}

    >>> import torch
    >>> results = torch.tensor(
    ...     [[0.5, 0.6, 0.7],
    ...      [0.4, 0.5, 0.6],
    ...      [0.3, 0.4, 0.5]])
    >>> expected = (0.5+0.5+0.5+0.4+0.4+0.3) / 6
    >>> actual = scenario_average(results)
    >>> print(f"{actual:.2f} == {expected:.2f}")
    0.43 == 0.43

    **References**:

    * Díaz-Rodríguez, N., Lomonaco, V., Filliat, D., & Maltoni, D. (2018).
      Don’t forget, there is more than forgetting: New metrics for Continual
      Learning (arXiv:1810.13166). arXiv. http://arxiv.org/abs/1810.13166

    :param results: A tensor of shape (``num_tasks``, ``num_tasks``) where
        ``results[i, j]`` is the performance of task ``j`` after training on
        task ``i``.
    :return: An average of the performance across all tasks.
    """
    num_tasks = _check_and_get_n_tasks(results)
    return torch.tril(results, diagonal=0).sum().item() / (
        num_tasks * (num_tasks + 1) / 2
    )


def seen_task_average(results: Tensor, train_task_id: int) -> float:
    r"""Average performance on past tasks after training on ``train_task_id``.

    Unlike :func:`scenario_average`, this metric does **NOT** take into account
    performance of the model at every time step but only performance after
    training on the given task.

    >>> import torch
    >>> results = torch.tensor(
    ...     [[0.5, 0.6, 0.7],
    ...      [0.4, 0.5, 0.6],
    ...      [0.3, 0.4, 0.5]])
    >>> expected = (0.4 + 0.5)/2
    >>> actual = seen_task_average(results, 1)
    >>> print(f"{actual:.2f} == {expected:.2f}")
    0.45 == 0.45

    :param results: A tensor of shape (``num_tasks``, ``num_tasks``) where
        ``results[i, j]`` is the performance of task ``j`` after training on
        task ``i``.
    :param train_task_id: The previous task used for training.
    :return: The forward transfer for the given task.
    """
    _check_and_get_n_tasks(results)
    return results[train_task_id, : train_task_id + 1].mean().item()


class CLMetric:
    """A class for managing a continual learning metric."""

    def __init__(self, num_tasks: int) -> None:
        """Construct a new CLMetric object.

        :param num_tasks: The number of tasks in the continual learning scenario.
        """
        self._num_tasks = num_tasks
        self._results = torch.zeros(num_tasks, num_tasks)
        self._seen = torch.zeros(num_tasks, num_tasks, dtype=torch.bool)

    def add(self, train_task_id: int, test_task_id: int, value: float | Tensor) -> None:
        """Add results to the results tensor.

        :param train_task_id: The task id of the task recently used for training.
        :param test_task_id: The task id of the test-set used to acquire the value.
        :param value: The value to add to the results tensor.
        """
        if isinstance(value, Tensor):
            value = value.item()
        self._results[train_task_id, test_task_id] = value
        self._seen[train_task_id, test_task_id] = True

    def _check_seen(self) -> None:
        if not self._seen.all():
            raise ValueError("Not all results have been added.")

    def forward_transfer(self) -> float:
        """See :func:`forward_transfer`."""
        self._check_seen()
        return forward_transfer(self._results)

    def backward_transfer(self) -> float:
        """See :func:`backward_transfer`."""
        self._check_seen()
        return backward_transfer(self._results)

    def scenario_average(self) -> float:
        """See :func:`cl_accuracy`."""
        self._check_seen()
        return scenario_average(self._results)

    def seen_task_average(self, train_task_id: int) -> float:
        """See :func:`seen_task_average`."""
        if not self._seen[train_task_id, : train_task_id + 1].all():
            raise ValueError("Not all results have been added.")
        return seen_task_average(self._results, train_task_id)
