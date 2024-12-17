import math
from typing import Generator, Iterator, List, Literal

import torch
from torch import BoolTensor, Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from clkan.model.base import HasTaskId
from clkan.model.wisemlp import WiseLinear
from clkan.model.wisenet import WiseNet

from .paraiso import (
    PrunableParameter,
    disable_grads,
    get_global_sparsity,
    prune_by_abs_magnitude,
    get_task_mask,
    get_task_sparsity,
    get_visible_mask,
)


class PnKANLinear(nn.Module, HasTaskId):
    def __init__(
        self,
        in_features,
        out_features,
        task_count: int,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1,
        scale_spline=1,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
        enable_standalone_scale_spline=True,
        enable_base_weight=True,
        norm: Literal["none", "batch", "layer"] = "none",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.task_id = 0

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.allocation = torch.nn.Parameter(
            torch.zeros(out_features * in_features, dtype=torch.uint8),
            requires_grad=False,
        )

        self.spline_weight = PrunableParameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features), requires_grad=enable_base_weight
        )
        if self.base_weight.requires_grad:
            self.base_weight.register_hook(
                lambda grad: disable_grads(self.allocation, grad, self.task_id)
            )

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
            self.spline_scaler.register_hook(
                lambda grad: disable_grads(self.allocation, grad, self.task_id)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.enable_base_weight = enable_base_weight
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

        self.norm = nn.Identity()
        self._batch_norm = False
        if norm == "batch":
            self._batch_norm = True
            # Each task maintains its own statistics to avoid interference.
            self.norm = nn.ModuleList(
                [nn.BatchNorm1d(in_features, affine=False) for _ in range(task_count)]
            )
        elif norm == "layer":
            self.norm = nn.LayerNorm(in_features, elementwise_affine=False, bias=False)
        else:
            assert norm == "none"

    @torch.no_grad()
    def reset_parameters(self):
        if self.enable_base_weight:
            torch.nn.init.kaiming_uniform_(
                self.base_weight, a=math.sqrt(5) * self.scale_base
            )
        else:
            self.base_weight.fill_(1)

        noise = (
            (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                - 1 / 2
            )
            * self.scale_noise
            / self.grid_size
        )
        self.spline_weight.parameter.data.copy_(
            (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
            * self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order],
                noise,
            )
        )
        if self.enable_standalone_scale_spline:
            # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
            torch.nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5) * self.scale_spline
            )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    def scaled_spline_weight(self, task_id: int):
        spline_scalar = 1.0
        if self.enable_standalone_scale_spline:
            spline_scalar = self.spline_scaler.unsqueeze(-1)
            spline_scalar = (
                spline_scalar.view(-1) * get_visible_mask(self.allocation, task_id)
            ).view_as(spline_scalar)

        return self.spline_weight.forward(task_id) * spline_scalar

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        if self._batch_norm:
            x = self.norm[self.task_id](x)
        else:
            x = self.norm(x)
        base_output = self._forward_base(x, self.task_id)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight(self.task_id).view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def _forward_base(self, x, task_id):
        if self.base_weight.requires_grad:
            base_weight = (
                self.base_weight.view(-1) * get_visible_mask(self.allocation, task_id)
            ).view_as(self.base_weight)
        else:
            base_weight = self.base_weight
        base_output = F.linear(self.base_activation(x), base_weight)
        return base_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        raise NotImplementedError("Not implemented yet")

    def regularization_loss(
        self, task_id: int, regularize_activation=1.0, regularize_entropy=1.0
    ):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        task_values = self.spline_weight.task_values(task_id)
        # It is possible that the task_values is empty if all edges are pruned.
        if task_values.numel() == 0:
            return torch.tensor(0.0, device=task_values.device)

        l1_fake = task_values.abs().sum() / self.spline_weight.parameter.numel()
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())

        assert not regularization_loss_activation.isnan().item()
        assert not regularization_loss_entropy.isnan().item()

        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        ).clamp_min(0)

    def spline_importance(self, task_id: int) -> Tensor:
        """Returns the mean scaled spline weight for each output unit."""
        return self.scaled_spline_weight(task_id).abs().mean(-1)

    @torch.no_grad()
    def inherit_allocation(self, task_id: int) -> None:
        """If a spline is pruned, the coefficients within that spline are also
        pruned.
        """
        mask = get_task_mask(self.allocation, task_id + 1).view(
            self.out_features, self.in_features, 1
        )
        sw_allocation = self.spline_weight.allocation.view(
            self.out_features, self.in_features, -1
        )
        sw_allocation.masked_fill_(mask, task_id + 1)

    def set_task_id(self, task_id: int) -> None:
        self.task_id = task_id

    def get_task_id(self) -> int:
        return self.task_id

    def count_parameters(self, task_id: int) -> int:
        total = 0
        if self.enable_base_weight:
            # Base weights are of shape (out_features, in_features)
            total += get_visible_mask(self.allocation, task_id).sum().item()
        if self.enable_standalone_scale_spline:
            # Spline scalers are of shape (out_features, in_features)
            total += get_visible_mask(self.allocation, task_id).sum().item()
        # Spline weights are of shape (out_features, in_features, grid_size + spline_order * 2)
        total += get_visible_mask(self.spline_weight.allocation, task_id).sum().item()
        return total


@torch.no_grad()
def _iterator_to_vector(iterator: Iterator[Tensor]) -> Tensor:
    return torch.cat([x.view(-1) for x in iterator])


class WiseKAN(WiseNet, nn.Module, HasTaskId):
    def __init__(
        self,
        layers_hidden,
        task_masks: BoolTensor,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        enable_standalone_scale_spline=True,
        enable_base_weight=True,
        first_layer_is_linear=False,
        last_layer_is_linear=False,
        norm: Literal["none", "batch", "layer"] = "none",
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.first_layer_is_linear = first_layer_is_linear
        self.last_layer_is_linear = last_layer_is_linear

        self.task_mask = nn.Parameter(task_masks, requires_grad=False)
        self.layers = torch.nn.Sequential()
        for layer_id, (in_features, out_features) in enumerate(
            zip(layers_hidden[:-1], layers_hidden[1:])
        ):
            if layer_id == 0 and first_layer_is_linear:
                self.layers.append(WiseLinear(in_features, out_features))
            elif layer_id == len(layers_hidden) - 2 and last_layer_is_linear:
                self.layers.append(WiseLinear(in_features, out_features))
            else:
                self.layers.append(
                    PnKANLinear(
                        in_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                        enable_base_weight=enable_base_weight,
                        enable_standalone_scale_spline=enable_standalone_scale_spline,
                        norm=norm,
                        task_count=len(task_masks),
                    )
                )

    def forward(self, x: torch.Tensor, task_id: int) -> Tensor:
        self.set_task_id(task_id, recursive=True)
        x = x.view(x.size(0), -1)
        return self.task_mask[task_id] * self.layers(x)

    def forward_all(self, x: torch.Tensor, task_ids: List[int] = None) -> Tensor:
        if task_ids is None:
            task_ids = list(range(len(self.task_mask)))
        return torch.stack([self(x, i) for i in task_ids], dim=0).permute(1, 0, 2)

    def forward_eval(self, x: torch.Tensor, task_ids: List[int] = None) -> Tensor:
        return self.forward_all(x, task_ids).sum(dim=1)

    def regularization_loss(
        self, task_id: int, regularize_activation=1.0, regularize_entropy=1.0
    ):
        reg_loss = sum(
            kan_linear.regularization_loss(
                task_id, regularize_activation, regularize_entropy
            )
            for kan_linear in self._iter_kan_linear()
        )
        if self.first_layer_is_linear:
            first_layer = self.layers[0]
            assert isinstance(first_layer, WiseLinear)
            reg_loss += first_layer.regularization_loss(task_id)
        if self.last_layer_is_linear:
            last_layer = self.layers[-1]
            assert isinstance(last_layer, WiseLinear)
            reg_loss += last_layer.regularization_loss(task_id)
        return reg_loss

    @torch.no_grad()
    def _iter_coef_importance(self, task_id: int):
        if self.first_layer_is_linear:
            first_layer = self.layers[0]
            assert isinstance(first_layer, WiseLinear)
            yield first_layer.weight.parameter
            yield first_layer.bias.parameter

        for module in self._iter_kan_linear():
            yield module.spline_weight.parameter

        if self.last_layer_is_linear:
            last_layer = self.layers[-1]
            assert isinstance(last_layer, WiseLinear)
            yield last_layer.weight.parameter
            yield last_layer.bias.parameter

    @torch.no_grad()
    def _iter_coef_allocation(self) -> Generator[Tensor, None, None]:
        if self.first_layer_is_linear:
            first_layer = self.layers[0]
            assert isinstance(first_layer, WiseLinear)
            yield first_layer.weight.allocation
            yield first_layer.bias.allocation

        for layer in self._iter_kan_linear():
            yield layer.spline_weight.allocation

        if self.last_layer_is_linear:
            last_layer = self.layers[-1]
            assert isinstance(last_layer, WiseLinear)
            yield last_layer.weight.allocation
            yield last_layer.bias.allocation

    @torch.no_grad()
    def _iter_kan_linear(self):
        for module in self.modules():
            if isinstance(module, PnKANLinear):
                if module.spline_weight.parameter.requires_grad:
                    yield module

    @torch.no_grad()
    def _iter_splines(self) -> Generator[PnKANLinear, None, None]:
        for module in self.modules():
            if isinstance(module, PnKANLinear):
                yield module

    @torch.no_grad()
    def _iter_spline_allocation(self) -> Generator[Tensor, None, None]:
        for module in self._iter_splines():
            yield module.allocation

    @torch.no_grad()
    def _iter_spline_importance(self, task_id: int) -> Generator[Tensor, None, None]:
        for module in self._iter_splines():
            yield module.spline_importance(task_id)

    @torch.no_grad()
    def spline_prune(self, task_id: int, sparsity: float, importance_override=None):
        if importance_override is not None:
            importance = importance_override
        else:
            importance = _iterator_to_vector(self._iter_spline_importance(task_id))

        allocations = _iterator_to_vector(self._iter_spline_allocation())
        assert importance.shape == allocations.shape
        vector_to_parameters(
            prune_by_abs_magnitude(allocations, importance, task_id, sparsity),
            self._iter_spline_allocation(),
        )
        for spline in self._iter_splines():
            spline.inherit_allocation(task_id)

    @torch.no_grad()
    def coef_prune(self, task_id: int, sparsity: float, importance_override=None):
        if importance_override is not None:
            importance = importance_override
        else:
            importance = _iterator_to_vector(self._iter_coef_importance(task_id))
        allocations = parameters_to_vector(self._iter_coef_allocation())
        vector_to_parameters(
            prune_by_abs_magnitude(allocations, importance, task_id, sparsity),
            self._iter_coef_allocation(),
        )

    def count_parameters(self, task_id: int) -> int:
        total = 0
        for module in self.modules():
            if isinstance(module, PnKANLinear):
                total += module.count_parameters(task_id)
            elif isinstance(module, WiseLinear):
                total += (
                    get_visible_mask(module.weight.allocation, task_id).sum().item()
                )
                if module.bias is not None:
                    total += (
                        get_visible_mask(module.bias.allocation, task_id).sum().item()
                    )
        return total

    @torch.no_grad()
    def coef_global_sparsity(self, unused_task_id: int) -> float:
        allocations = parameters_to_vector(self._iter_coef_allocation())
        return get_global_sparsity(allocations, unused_task_id)

    @torch.no_grad()
    def coef_task_sparsity(self, task_id: int) -> float:
        allocations = parameters_to_vector(self._iter_coef_allocation())
        return get_task_sparsity(allocations, task_id)

    @torch.no_grad()
    def spline_global_sparsity(self, unused_task_id: int) -> float:
        allocations = parameters_to_vector(self._iter_spline_allocation())
        return get_global_sparsity(allocations, unused_task_id)

    @torch.no_grad()
    def spline_task_sparsity(self, task_id: int) -> float:
        allocations = parameters_to_vector(self._iter_spline_allocation())
        return get_task_sparsity(allocations, task_id)
