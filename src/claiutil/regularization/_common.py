"""Implements common components for regularization strategies."""

from typing import Iterable, Iterator, Sequence, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, Parameter, ParameterList

MutTensorSequence = TypeVar("MutTensorSequence", Sequence[Tensor], ParameterList)
"""A sequence of tensors which can be mutated in-place."""


__T_only_learnable = TypeVar("__T_only_learnable", Tensor, Parameter)


def _only_learnable(
    tensors: Iterable[__T_only_learnable],
) -> Iterator[__T_only_learnable]:
    """Filter out tensors that do not require gradients.

    :param tensors: The tensors to filter.
    :return: An iterable of tensors that require gradients.
    """
    return filter(lambda tensor: tensor.requires_grad, tensors)


def parameters_like_module(
    module: Module, only_learnable: bool = True
) -> ParameterList:
    """Create a new sequence of parameters with the same shape as the given module.

    * The new parameters are initialized to zero. You can use :func:`copy_parameters`
        to copy the parameters from another module.
    * The new parameters are not learnable e.g ``requires_grad`` is False.

    >>> from torch import nn
    >>> no_grad_module = nn.Linear(2, 2).requires_grad_(False)
    >>> module = nn.Sequential(nn.Linear(2, 2), no_grad_module, nn.Linear(2, 2))
    >>> parameters = parameters_like_module(module, only_learnable=True)
    >>> all(param.requires_grad for param in parameters)
    False
    >>> parameters
    ParameterList(
        (0): Parameter containing: [torch.float32 of size 2x2]
        (1): Parameter containing: [torch.float32 of size 2]
        (2): Parameter containing: [torch.float32 of size 2x2]
        (3): Parameter containing: [torch.float32 of size 2]
    )

    :param module: The module to copy the parameters from.
    :param only_learnable: If True, only parameters that require gradients are copied.
    :return: A sequence of tensors with the same shape as the module's parameters.
    """
    src = module.parameters()
    src = _only_learnable(src) if only_learnable else src
    return ParameterList(torch.zeros_like(param) for param in src).requires_grad_(False)


@torch.no_grad()
def copy_parameters(
    dst: MutTensorSequence, src: Iterable[Tensor], only_learnable: bool = True
) -> MutTensorSequence:
    """Copy the parameters from one sequence to another in-place.

    :param dst: The destination sequence to copy the parameters to.
    :param src: The source sequence to copy the parameters from.
    :param only_learnable: If True, only parameters that require gradients are
        copied.
    :return: The destination sequence.
    """
    src = _only_learnable(src) if only_learnable else src
    for i, (src_p, _) in enumerate(zip(src, dst, strict=True)):
        dst[i].copy_(src_p.data)
    return dst


def zero_parameters(parameters: MutTensorSequence) -> MutTensorSequence:
    """Zero the sequence of tensors in-place."""
    for param in parameters:
        param.zero_()
    return parameters


def _all_close(
    a: Iterable[Tensor], b: Iterable[Tensor], atol: float = 1e-8, rtol: float = 1e-5
) -> bool:
    for a_p, b_p in zip(a, b, strict=True):
        if not torch.allclose(a_p.data, b_p.data.view_as(a_p), atol=atol, rtol=rtol):
            return False
    return True


def quadratic_penalty(
    model_params: Iterable[Tensor],
    target_model_params: Iterable[Tensor],
    importances: Iterable[Tensor],
    only_learnable: bool = True,
) -> Tensor:
    r"""Calculate a penalty for regularization based on parameter importance.

    The EWC penalty is the sum of the importance of each parameter multiplied by
    the square of the difference between the current parameter and the parameter
    in the target model.

    :param model_params: The parameters of the model. This should be the same
        shape as ``target_model_params``. See :meth:`nn.Module.parameters`.
    :param target_model_state_dict: The parameters that the model should try to
        match. This is usually the parameters of the model before training on
        the current task. This should be the same shape as ``model_params``. See
        :meth:`nn.Module.state_dict`.
    :param importances: The importance of each parameter in the model. This
        should be the same shape as ``model_params``. See
        :func:`ewc_fisher_importance` or :func:`si_update_importances` to
        calculate these importances.
    :param only_learnable: If True, ``model_params`` is filtered to only include
        parameters that require gradients. This does not affect
        ``pre_task_params``, ``trajectory``, or ``importances``, which should be
        the same shape as the filtered ``model_params``.
    :return: A scalar penalty for regularization.
    """
    penalty = torch.tensor(0.0)
    for param, target_param, importance in zip(
        _only_learnable(model_params) if only_learnable else model_params,
        target_model_params,
        importances,
        strict=True,
    ):
        # Why the 1/2 factor? It is to cancel out the 2 in the derivative of the square
        # https://math.stackexchange.com/a/884903
        penalty = penalty.to(param.device)
        penalty += 1 / 2 * (importance * (param - target_param) ** 2).sum()
    return penalty
