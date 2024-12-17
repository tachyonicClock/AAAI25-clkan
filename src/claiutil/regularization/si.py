"""Implements Synaptic Intelligence (SI) [Zenke17]_.

.. [Zenke17] Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning
    Through Synaptic Intelligence. Proceedings of the 34th International
    Conference on Machine Learning, 3987–3995.
    http://proceedings.mlr.press/v70/zenke17a.html

"""

from typing import Iterable

import torch
from torch import Tensor

from ._common import MutTensorSequence, _only_learnable


@torch.no_grad()
def si_update_trajectory(
    trajectory: MutTensorSequence,
    model_params: Iterable[Tensor],
    pre_step_params: Iterable[Tensor],
    only_learnable: bool = True,
) -> MutTensorSequence:
    r"""Update the parameter's cumulative trajectory for Synaptic Intelligence.

    Should be called after optimizer step but before zeroing the gradients. For
    example in PyTorch Lightning::

        def optimizer_step(self, *args, **kwargs) -> None:
            super().optimizer_step(*args, **kwargs)
            self.on_after_optimizer_step() # Call after optimizer step

        def on_after_optimizer_step(self) -> None:
            self.trajectory = si_update_trajectory(...)

    This function implements Equation 2 from [Zenke17]_:

    ..  math::

        \begin{aligned} \int_{t^{\mu-1}}^{t^\mu}
        \boldsymbol{g}(\boldsymbol{\theta}(t)) \cdot
        \boldsymbol{\theta}^{\prime}(t) d t & =\sum_k \int_{t^{\mu-1}}^{t^\mu}
        g_k(\theta(t)) \theta_k^{\prime}(t) d t \\ & \equiv-\sum_k \omega_k^\mu,
        \end{aligned}

    where:

    *   :math:`\boldsymbol{g}(\boldsymbol{\theta}(t))` is the gradient of the
        loss with respect to the parameters at optimization step :math:`t`.
    *   :math:`\boldsymbol{\theta}^{\prime}(t)` is the difference between the
        parameters at optimization step :math:`t` (``model_params``) and the
        parameters at :math:`t-1` (``pre_step_params``).

    This function updates the trajectory :math:`\omega_k^\mu` for each parameter
    :math:`k` after each optimization step.

    :param trajectory: The trajectory to update. **This is updated in-place.**
    :param model_params: The parameters of the model. These should have
        gradients left over from the backwards pass. This should be the same
        shape as ``trajectory`` (unless ``only_learnable`` is True).
    :param pre_step_params: The parameters of the model before the step. This
        should be the same shape as ``trajectory``.
    :param only_learnable: If True, ``model_params`` is filtered to only include
        parameters that require gradients. This does not affect ``pre_step_params``,
        or ``trajectory``, which should be the same shape as the filtered
        ``model_params``.
    :return: The updated trajectory
    """
    any_grads = False
    for i, (param, pre_param, _) in enumerate(
        zip(
            _only_learnable(model_params) if only_learnable else model_params,
            pre_step_params,
            trajectory,
            strict=True,
        )
    ):
        if param.grad is not None:
            any_grads = True
            # Equation 3 has a negative sign, so we subtract the gradient
            # This is sometimes done elsewhere in other implementations
            trajectory[i].sub_(param.grad * (param - pre_param))

    if not any_grads:
        raise ValueError("No gradients found in model parameters")

    return trajectory


@torch.no_grad()
def si_update_importances(
    importances: MutTensorSequence,
    model_params: Iterable[Tensor],
    pre_task_params: Iterable[Tensor],
    trajectory: Iterable[Tensor],
    epsilon: float = 0.0000001,
    only_learnable: bool = True,
) -> MutTensorSequence:
    r"""Calculate the importance of each parameter using Synaptic Intelligence.

    * It should be called to update the importance of each parameter after each
      task.
    * The ``importances`` argument is updated in-place. You should make a copy
      if you want to keep the previous importances.
    * Implements Equation 5 from [Zenke17]_:

        ..  math::

            \Omega_k^\mu=\sum_{\nu<\mu}
            \frac{\omega_k^\nu}{\left(\Delta_k^\nu\right)^2+\xi}

        Rather than calculate the right hand side sum in one go, we accumulate
        it by updating the ``importances`` after each task.

        * :math:`\Delta_k^\nu` is calculated using ``model_params`` and
          ``pre_task_params``.
        * :math:`\omega_k^\nu` is the cumulative trajectory of the model for a
          single task. and is named ``trajectory`` in this function. Use
          :func:`si_trajectory_update` to accumulate it.
        * :math:`\xi` is a small ``epsilon`` value to prevent division by zero.
    * Use the importances with :func:`claiutil.regularization.quadratic_penalty`
      to calculate the penalty.

    :param importances: The importance of each parameter in the model. **This is
        updated in-place.**
    :param model_params: The parameters of the model. This should be the same
        shape as ``importances`` (unless ``only_learnable`` is True).
    :param pre_task_params: The parameters of the model before the task. This
        should be the same shape as ``importances``.
    :param trajectory: The cumulative trajectory of the model for a single task.
        This should be the same shape as ``importances``.
    :param epsilon: A small value to prevent division by zero.
    :param only_learnable: If True, ``model_params`` is filtered to only include
        parameters that require gradients. This does not affect
        ``pre_task_params``, ``trajectory``, or ``importances``, which should be
        the same shape as the filtered ``model_params``.
    :return: A sequence of tensors, each representing the importance of a
        parameter.
    """
    for i, (param, pre_param, traj, _) in enumerate(
        zip(
            _only_learnable(model_params) if only_learnable else model_params,
            pre_task_params,
            trajectory,
            importances,
            strict=True,
        )
    ):
        delta = param - pre_param
        importances[i].add_(traj / (delta**2 + epsilon))
    return importances
