"""Implements Elastic Weight Consolidation (EWC) [Kirkpatrick2017]_.

..  seealso::

    * :func:`claiutil.regularization.quadratic_penalty` to calculate
      penalties for regularization.
    * :func:`claiutil.regularization.parameters_like_module` to initialize
      sequences of tensors representing parameter importances.

.. [Kirkpatrick2017] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J.,
    Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T.,
    Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R.
    (2017). Overcoming catastrophic forgetting in neural networks. Proceedings
    of the National Academy of Sciences, 114(13), 3521–3526.
    https://doi.org/10.1073/pnas.1611835114
"""

from typing import Callable, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from claiutil._util import evaluation

from ._common import MutTensorSequence, _only_learnable


@torch.no_grad()
def ewc_fisher_update_importances(
    importances: MutTensorSequence,
    model: nn.Module,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    device: torch.device | str,
    only_learnable: bool = True,
) -> MutTensorSequence:
    """Calculate the importance of each parameter in the model for EWC.

    The importance in EWC is defined as the diagonal of the Fisher Information
    Matrix.

    :param model: The model to calculate the importance for.
    :param loss_func: The loss function used to calculate the loss.
    :param dataloader: A dataset used to calculate the importance.
    :param only_learnable: If True, only parameters that require gradients are
        have their importance calculated.
    :return: A sequence of tensors, each representing the importance of a
        parameter
    """
    num_batches = len(dataloader)

    with evaluation(model) as model, torch.enable_grad():  # type: ignore
        for batch in dataloader:
            inputs: Tensor = batch[0].to(device)
            targets: Tensor = batch[1].to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()  # type: ignore
            for i, (param, _) in enumerate(
                zip(
                    (
                        _only_learnable(model.parameters())
                        if only_learnable
                        else model.parameters()
                    ),
                    importances,
                    strict=True,
                )
            ):
                if param.grad is not None:
                    importances[i].add_(param.grad.detach() ** 2)

    # Average the fisher information matrix over all batches
    for i, _ in enumerate(importances):
        importances[i].div_(num_batches)

    return importances
