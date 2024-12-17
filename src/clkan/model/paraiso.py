import math

import torch
from torch import BoolTensor, ByteTensor, LongTensor, Tensor, nn


def get_visible_mask(allocation: ByteTensor, task_id: int) -> BoolTensor:
    """Get a binary mask for the parameters visible to the given task.

    Parameters from the given task and all previous tasks are visible.

    >>> allocation = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.uint8)
    >>> get_visible_mask(allocation, 0)
    tensor([ True,  True, False, False, False, False, False, False])
    >>> get_visible_mask(allocation, 1)
    tensor([ True,  True,  True,  True, False, False, False, False])
    >>> get_visible_mask(allocation, 2)
    tensor([ True,  True,  True,  True,  True,  True, False, False])

    :param allocation: The allocation of parameters to tasks.
    :param task_id: The task ID.
    :return: A binary mask for the parameters visible to the given task.
    """
    return (allocation <= task_id).bool()


def get_task_mask(allocation: ByteTensor, task_id: int) -> BoolTensor:
    """Get a binary mask for the parameters allocated to the given task.

    Note that this mask is not the same as the visible mask.

    >>> allocation = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
    >>> task_mask(allocation, 0)
    tensor([ True,  True, False, False])
    >>> task_mask(allocation, 1)
    tensor([False, False,  True,  True])

    :param allocation: The allocation of parameters to tasks.
    :param task_id: The task ID.
    :return: A binary mask for the parameters allocated to the given task.
    """
    return (allocation == task_id).bool()


def get_global_sparsity(allocation: ByteTensor, unused_task_id: int) -> float:
    """Calculate the percentage of parameters that are not allocated to any task.

    :param allocation: The allocation of parameters to tasks.
    :return: The global sparsity.
    """
    count = get_task_mask(allocation, unused_task_id).sum().item()
    return count / allocation.numel()


def get_task_sparsity(allocation: ByteTensor, task_id: int) -> float:
    """Calculate the percentage of parameters that are not allocated to the given task.

    Returns 0 if the task has no parameters.

    :param allocation: The allocation of parameters to tasks.
    :param task_id: The task ID to calculate the sparsity for.
    :param unused_task_id: The task ID that represents unallocated parameters.
    :return: The task sparsity.
    """
    assert allocation.ndim == 1, "`allocation` must be a 1D tensor"
    num_unused = (allocation > task_id).sum().item()
    num_used = (allocation == task_id).sum().item()
    return num_unused / (num_unused + num_used) if (num_unused + num_used) != 0 else 0


def disable_grads(allocation: ByteTensor, grads: Tensor, task_id: int) -> Tensor:
    return (grads.view(-1) * (allocation == task_id)).view_as(grads)


def update_allocation(
    allocation: ByteTensor,
    src_prune_order: LongTensor,
    target_src_sparsity: float,
    src_task_id: int,
    dst_task_id: int,
) -> ByteTensor:
    """Update the allocation of parameters to tasks by moving the least
    important parameters from the source task to the destination task.

    The number of parameters to move is determined by the target sparsity
    ``src_sparsity``.

    >>> allocation = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)
    >>> src_prune_order = torch.tensor([0, 1, 2, 3, 4, 5])
    >>> target_src_sparsity = 0.5
    >>> update_allocation(allocation, src_prune_order, target_src_sparsity, 0, 1)
    tensor([1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)
    >>> update_allocation(allocation, torch.tensor([0, 1, 2, 3]), target_src_sparsity, 0, 1)
    tensor([1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)
    >>> update_allocation(allocation, torch.tensor([0, 1, 2, 3]), 0.75, 0, 1)
    tensor([1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.uint8)



    :param allocation: The allocation of parameters to tasks. Updated in-place.
    :param src_indices: A tensor of indices of the source task parameters sorted by importance.
        The least important parameters are moved to the destination task.
    :param target_src_sparsity: The sparsity of the source task after the parameters are moved.
    :param src_task_id: The task ID that donates the parameters to the destination task.
    :param dst_task_id: The task ID that receives the parameters from the source task.
    :return: The updated allocation.
    """
    allocation = allocation.view(-1)
    src_prune_order = src_prune_order.view(-1)
    assert 0 <= target_src_sparsity <= 1, "The sparsity must be between 0 and 1"
    if src_prune_order.numel() == 0:
        return allocation

    src_parameters = get_task_mask(allocation, src_task_id).sum().item()
    dst_parameters = get_task_mask(allocation, dst_task_id).sum().item()
    assert (
        len(src_prune_order) == src_parameters
    ), f"The number of indices ({len(src_prune_order)}) must match the number of parameters in the source task ({src_parameters})"
    assert (
        0 <= src_prune_order.max() < src_parameters
    ), "The indices must be in the range of the source task parameters"

    total_parameters = src_parameters + dst_parameters

    # Calculate the number of parameters to move
    target_count = math.ceil(total_parameters * (1 - target_src_sparsity))
    num_parameters = src_parameters - target_count
    if num_parameters < 0:
        return allocation

    # Update the allocation
    task_mask = get_task_mask(allocation, src_task_id)
    allocation[task_mask] = allocation[task_mask].index_fill(
        0, src_prune_order[:num_parameters], dst_task_id
    )
    return allocation


def prune_by_abs_magnitude(
    allocations: ByteTensor, parameters: Tensor, task_id: int, sparsity: float
) -> Tensor:
    task_parameters = parameters[get_task_mask(allocations, task_id)]
    importance = torch.argsort(task_parameters.abs(), descending=False)
    return update_allocation(allocations, importance, sparsity, task_id, task_id + 1)


class PrunableParameter(nn.Module):
    def __init__(self, parameter: Tensor):
        super().__init__()
        self.parameter = nn.Parameter(parameter, requires_grad=True)
        self._last_task_id = 0
        self.allocation = nn.Parameter(
            torch.zeros(parameter.numel(), dtype=torch.uint8), requires_grad=False
        )
        if self.parameter.requires_grad:
            self.parameter.register_hook(
                lambda grad: disable_grads(self.allocation, grad, self._last_task_id)
            )

    def forward(self, task_id: int) -> Tensor:
        if self.parameter.requires_grad:
            self._last_task_id = task_id
            return (
                self.parameter.view(-1) * get_visible_mask(self.allocation, task_id)
            ).view_as(self.parameter)
        return self.parameter

    def task_values(self, task_id: int) -> Tensor:
        return self.parameter.view(-1)[get_task_mask(self.allocation, task_id)]
