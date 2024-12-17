import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import init
from torch.nn.functional import linear
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from clkan.model.base import HasTaskId
from clkan.model.paraiso import (
    PrunableParameter,
    get_global_sparsity,
    prune_by_abs_magnitude,
    get_task_sparsity,
    get_visible_mask,
)
from clkan.model.wisenet import WiseNet


class WiseLinear(nn.Module, HasTaskId):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = PrunableParameter(Tensor(out_features, in_features))
        self.bias = PrunableParameter(Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight.parameter, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight.parameter)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias.parameter, -bound, bound)

    def forward(self, x: Tensor, task_id: int = None) -> Tensor:
        if task_id is not None:
            self.set_task_id(task_id)
        return linear(
            x,
            self.weight.forward(self.get_task_id()),
            self.bias.forward(self.get_task_id()),
        )

    def regularization_loss(self, task_id: int) -> Tensor:
        weight_values = self.weight.task_values(task_id)
        bias_values = self.bias.task_values(task_id)
        loss = 0
        if weight_values.numel() > 0:
            loss += weight_values.abs().mean()
        if bias_values.numel() > 0:
            loss += bias_values.abs().mean()
        return loss


class PackNet(WiseNet, nn.Module):
    def __init__(self, widths: Sequence[int], task_mask: Tensor):
        super(PackNet, self).__init__()
        self.task_mask = nn.Parameter(task_mask, requires_grad=False)
        self.layers = nn.ModuleList(
            [WiseLinear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
        )
        self.act_fn = nn.ReLU()

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        # Forward pass through all layers except the last one
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            layer: WiseLinear
            x = layer.forward(x, task_id)
            x = self.act_fn(x)

        # Forward pass through the last layer
        return self.task_mask[task_id] * self.layers[-1].forward(x, task_id)

    def forward_all(self, x: torch.Tensor, task_ids: List[int] = None) -> Tensor:
        if task_ids is None:
            task_ids = list(range(len(self.task_mask)))
        return torch.stack([self(x, i) for i in task_ids], dim=0).permute(1, 0, 2)

    def forward_eval(self, x: torch.Tensor, task_ids: List[int] = None) -> Tensor:
        return self.forward_all(x, task_ids).softmax(dim=2).sum(dim=1)

    @torch.no_grad()
    def _iter_prunable_parameters(self):
        for modules in self.modules():
            if isinstance(modules, PrunableParameter):
                yield modules

    @torch.no_grad()
    def _iter_parameters(self):
        for prunable_parameter in self._iter_prunable_parameters():
            yield prunable_parameter.parameter

    @torch.no_grad()
    def _iter_allocation(self):
        for prunable_parameter in self._iter_prunable_parameters():
            yield prunable_parameter.allocation

    def regularization_loss(
        self, task_id: int, regularize_activation=1.0, regularize_entropy=1.0
    ) -> Tensor:
        raise NotImplementedError("Regularization loss is not supported for MLPs")

    def spline_prune(self, task_id: int, sparsity: float):
        raise NotImplementedError("Spline pruning is not supported for MLPs")

    def coef_prune(
        self,
        task_id: int,
        sparsity: float,
        coef_importance_override: Optional[Tensor] = None,
    ):
        allocations = parameters_to_vector(self._iter_allocation())
        importance = coef_importance_override is not None or parameters_to_vector(
            self._iter_parameters()
        )
        vector_to_parameters(
            prune_by_abs_magnitude(allocations, importance, task_id, sparsity),
            self._iter_allocation(),
        )

    def count_parameters(self, task_id: int) -> int:
        allocations = parameters_to_vector(self._iter_allocation())
        return get_visible_mask(allocations, task_id).sum().item()

    def coef_global_sparsity(self, unused_task_id: int) -> float:
        allocations = parameters_to_vector(self._iter_allocation())
        return get_global_sparsity(allocations, unused_task_id)

    def coef_task_sparsity(self, task_id: int) -> float:
        allocations = parameters_to_vector(self._iter_allocation())
        return get_task_sparsity(allocations, task_id)

    def spline_global_sparsity(self, unused_task_id: int) -> float:
        return None

    def spline_task_sparsity(self, task_id: int) -> float:
        return None
