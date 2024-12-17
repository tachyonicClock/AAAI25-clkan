import io
from itertools import product
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from clkan.model.wisekan import PnKANLinear, WiseKAN, WiseNet
from efficient_kan import KAN as EfficientKAN
from efficient_kan import KANLinear as EfficientKANLinear
from kan import KAN as PyKAN
from kan import KANLayer as PyKANLayer
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

PLT_FIGURE_KWARGS = dict(
    figsize=(1, 1),
    dpi=300,
    layout="tight",
)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class PyKANPlot:
    ax: Axes
    spline_width = 1.0
    subplot_figsize = (2, 2)
    height_multiplier = 1.2
    horizontal: bool = False

    def __init__(self, kan_width: List[int], horizontal: bool = False) -> None:
        self.kan_width = kan_width
        self.horizontal = horizontal

        num_rows = self.kan_layers * 2 - 1  # The rows include the nodes and the edges

        self._nodes: List[int] = []
        for plot_group in range(self.kan_layers):
            self._nodes.append(self.kan_width[plot_group])
            if plot_group + 1 < self.kan_layers:
                self._nodes.append(
                    self.kan_width[plot_group] * self.kan_width[plot_group + 1]
                )

        self.width = max(self._nodes)
        figsize = (self.width, num_rows * self.height_multiplier)
        if self.horizontal:
            figsize = (figsize[1], figsize[0])

        self.fig, self.ax = plt.subplots(figsize=figsize)
        if self.horizontal:
            self.ax.set_ylim(0, self.width)
            self.ax.set_xlim(0, num_rows * self.height_multiplier)
        else:
            self.ax.set_ylim(0, num_rows * self.height_multiplier)
            self.ax.set_xlim(0, self.width)
        self.ax.set_axis_off()
        self.ax.set_aspect("equal")

        self._kwargs_plot_edge = dict(color="black", zorder=0)
        self._kwargs_scatter_spline = dict(color="red", s=100, zorder=10)
        self._kwargs_scatter_neuron = dict(color="black", s=100)

    @property
    def kan_layers(self):
        return len(self.kan_width)

    def _to_horizontal(self, x, y):
        x = self.width - x
        return y, x

    def _neuron_pos(self, layer: int, node: int) -> Tuple[float, float]:
        y = (layer * 2 + 0.5) * self.height_multiplier
        x = (node + 0.5) * self.width / self.kan_width[layer]
        if self.horizontal:
            x, y = self._to_horizontal(x, y)
        return x, y

    def _spline_pos(
        self, layer: int, in_node: int, out_node: int
    ) -> Tuple[float, float]:
        y = (layer * 2 + 1.5) * self.height_multiplier
        spline_idx = in_node * self.kan_width[layer + 1] + out_node
        x = (
            (spline_idx + 0.5)
            * self.width
            / (self.kan_width[layer] * self.kan_width[layer + 1])
        )
        if self.horizontal:
            x, y = self._to_horizontal(x, y)
        return x, y

    def _iterate_splines(self):
        for layer in range(self.kan_layers - 1):
            for in_node, out_node in product(
                range(self.kan_width[layer]), range(self.kan_width[layer + 1])
            ):
                yield layer, in_node, out_node

    def _iterate_neurons(self):
        for layer in range(self.kan_layers):
            for i in range(self.kan_width[layer]):
                yield layer, i

    def draw_splines(self):
        for layer, in_node, out_node in self._iterate_splines():
            x, y = self._spline_pos(layer, in_node, out_node)
            self.ax.scatter(x, y, **self._kwargs_scatter_spline)

    def draw_neurons(self):
        for layer, node in self._iterate_neurons():
            x, y = self._neuron_pos(layer, node)
            self.ax.scatter(x, y, **self._kwargs_scatter_neuron)

    def draw_edges(self):
        for layer, in_node, out_node in self._iterate_splines():
            x, y = self._spline_pos(layer, in_node, out_node)
            nx, ny = self._neuron_pos(layer, in_node)
            px, py = self._neuron_pos(layer + 1, out_node)
            self.ax.plot([nx, x], [ny, y], **self._kwargs_plot_edge)
            self.ax.plot([x, px], [y, py], **self._kwargs_plot_edge)

    def add_spline_figures(
        self, plot_spline: Callable[[int, int, int, dict], Image.Image]
    ):
        for layer, i, j in self._iterate_splines():
            x, y = self._spline_pos(layer, i, j)
            spline = plot_spline(layer, i, j, {})
            w = self.spline_width / 2
            extent = (x - w, x + w, y - w, y + w)
            self.ax.imshow(spline, extent=extent, zorder=10)
            self.ax.annotate(
                f"{layer},{i},{j}",
                (x + (0.2 * w), y + (0.85 * w)),
                zorder=20,
                ha="center",
            )

    def _plot_activation_distribution(
        self,
        sample_outputs: List[torch.Tensor],
        grid_range: Tuple[float, float],
        hide_ticks: bool = False,
        dpi: int = 300,
    ) -> Image.Image:
        with plt.ioff():
            fig, ax = plt.subplots(**PLT_FIGURE_KWARGS)
            ax: Axes
            sample_outputs = list(map(lambda x: x.cpu().numpy(), sample_outputs))

            x = np.concatenate(sample_outputs)
            bins = np.histogram_bin_edges(x, bins="auto")

            for x in sample_outputs:
                ax.hist(x, bins=bins, range=grid_range, density=True, alpha=0.75)

            ax.yaxis.set_visible(False)
            ax.set_box_aspect(1)

            ax.set_xticks(grid_range)

            if hide_ticks:
                ax.set_yticks([])
                ax.set_xticks([])

            image = fig2img(fig)
            plt.close(fig)
            return image

    def add_activations(
        self,
        activations: List[List[torch.Tensor]],
        grid_range: Tuple[float, float],
        hide_ticks: bool = False,
        dpi: int = 300,
    ):
        for layer, in_node in self._iterate_neurons():
            x, y = self._neuron_pos(layer, in_node)
            task_activations = list(map(lambda x: x[layer][:, in_node], activations))
            activation = self._plot_activation_distribution(
                task_activations, grid_range, hide_ticks=hide_ticks, dpi=dpi
            )
            w = self.spline_width / 2
            extent = (x - w, x + w, y - w, y + w)
            self.ax.imshow(activation, extent=extent, zorder=10)


def efficient_kan_spline(i, j, kan_linear, x_batch):
    b_spline_out = kan_linear.b_splines(x_batch)
    coefficients = kan_linear.scaled_spline_weight[j, i]
    y = b_spline_out[:, i] @ coefficients
    base_output = F.linear(kan_linear.base_activation(x_batch), kan_linear.base_weight)
    y = y + base_output[:, j]
    return y


def pnkan_spline(i, j, kan_linear: PnKANLinear, x_batch, task_id):
    b_spline_out = kan_linear.b_splines(x_batch)
    coefficients = kan_linear.scaled_spline_weight(task_id)[j, i]
    base_output = kan_linear._forward_base(x_batch, task_id)
    y = b_spline_out[:, i] @ coefficients
    y = y + base_output[:, j]
    return y


def efficient_kan_plot_spline(
    model: EfficientKAN,
    resolution: int = 100,
    dpi: int = 300,
    grid_range: Optional[Tuple[float, float]] = None,
):
    @torch.no_grad()
    def plot_spline(layer: int, i: int, j: int, plot_kwargs: dict) -> Image.Image:
        kan_linear: EfficientKANLinear = model.layers[layer]

        grid_max = max(kan_linear.grid[i])
        grid_min = min(kan_linear.grid[i])
        if grid_range is not None:
            grid_min, grid_max = grid_range
        x = torch.linspace(
            grid_min, grid_max, resolution, device=kan_linear.grid.device
        )
        x_batch = torch.zeros(
            (resolution, kan_linear.in_features), device=kan_linear.grid.device
        )
        x_batch[:, i] = x

        # Shape: (resolution, num_splines, knots)
        y = efficient_kan_spline(i, j, kan_linear, x_batch)

        with plt.ioff():
            fig, ax = plt.subplots(**PLT_FIGURE_KWARGS)
            ax: Axes
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), color="black", **plot_kwargs)
            image = fig2img(fig)
            plt.close(fig)
            return image

    return plot_spline


def pnkan_plot_splines(model: WiseKAN, task_ids: List[int], resolution: int = 100):
    @torch.no_grad()
    def plot_spline(layer: int, i: int, j: int, plot_kwargs: dict) -> Image.Image:
        kan_linear: PnKANLinear = model.layers[layer]

        grid_max = max(kan_linear.grid[i])
        grid_min = min(kan_linear.grid[i])
        x = torch.linspace(
            grid_min, grid_max, resolution, device=kan_linear.grid.device
        )
        x_batch = torch.zeros(
            (resolution, kan_linear.in_features), device=kan_linear.grid.device
        )
        x_batch[:, i] = x

        with plt.ioff():
            fig, ax = plt.subplots(**PLT_FIGURE_KWARGS)
            ax: Axes
            ax.set_box_aspect(1)

            for task_id in task_ids:
                y = pnkan_spline(i, j, kan_linear, x_batch, task_id)
                ax.plot(x.cpu().numpy(), y.cpu().numpy(), **plot_kwargs)
            image = fig2img(fig)
            plt.close(fig)
            return image

    return plot_spline


def plot_two_kans(
    model: EfficientKAN,
    old_model: Optional[EfficientKAN],
    resolution: int = 100,
    grid_range: Optional[Tuple[float, float]] = None,
    hide_ticks: bool = False,
):
    model.cpu()
    old_model is None or old_model.cpu()

    @torch.no_grad()
    def plot_spline(layer: int, i: int, j: int, plot_kwargs: dict) -> Image.Image:
        kan_linear: EfficientKANLinear = model.layers[layer]

        if grid_range is not None:
            grid_min, grid_max = grid_range
        else:
            grid_max = max(kan_linear.grid[i])
            grid_min = min(kan_linear.grid[i])
        x = torch.linspace(
            grid_min, grid_max, resolution, device=kan_linear.grid.device
        )
        x_batch = torch.zeros(
            (resolution, kan_linear.in_features), device=kan_linear.grid.device
        )
        x_batch[:, i] = x

        # Shape: (resolution, num_splines, knots)
        y = efficient_kan_spline(i, j, kan_linear, x_batch)

        with plt.ioff():
            fig, ax = plt.subplots(**PLT_FIGURE_KWARGS)
            ax: Axes
            if old_model is not None:
                y_old = efficient_kan_spline(i, j, old_model.layers[layer], x_batch)
                ax.plot(
                    x.cpu().numpy(),
                    y_old.cpu().numpy(),
                    alpha=0.5,
                    **plot_kwargs,
                )
                ax.plot(x.cpu().numpy(), y.cpu().numpy(), **plot_kwargs)
            else:
                ax.plot(x.cpu().numpy(), y.cpu().numpy(), color="black")
            ax.set_box_aspect(1)

            if hide_ticks:
                ax.set_yticks([])
                ax.set_xticks([])
                # Horizontal line at 0
                ax.axhline(0, color="grey", linewidth=1)

            image = fig2img(fig)
            plt.close(fig)
            return image

    return plot_spline


def pykan_plot_spline(model: PyKAN, resolution: int = 100):
    @torch.no_grad()
    def plot_spline(
        layer: int, i: int, j: int, plot_kwargs: dict = dict()
    ) -> Image.Image:
        kan_layer: PyKANLayer = model.act_fun[layer]

        grid_max = kan_layer.grid[i].max()
        grid_min = kan_layer.grid[i].min()

        x = torch.linspace(grid_min, grid_max, resolution, device=kan_layer.device)
        x_batch = torch.zeros((resolution, kan_layer.in_dim), device=kan_layer.device)
        x_batch[:, i] = x
        _, _, postacts, _ = kan_layer.forward(x_batch)
        y = postacts[:, j, i]

        with plt.ioff():
            fig, ax = plt.subplots(figsize=(2, 2), layout="tight")
            ax: Axes
            # import pdb
            # pdb.set_trace()
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), color="black", **plot_kwargs)
            image = fig2img(fig)
            plt.close(fig)
            return image

    return plot_spline


@torch.no_grad()
def pykan_activations(model: PyKAN, x: torch.Tensor) -> List[torch.Tensor]:
    activations = []
    removable_hooks = []

    activations.append(x)

    # Register Hooks
    for kan_layer in model.act_fun:
        kan_layer: PyKANLayer

        def hook_track_activation(module, input, output):
            activations.append(output[0])

        removable_hooks.append(kan_layer.register_forward_hook(hook_track_activation))

    # Forward Pass
    model(x.to(model.device))

    # Remove All Hooks
    for _hook in removable_hooks:
        _hook.remove()

    return activations


@torch.no_grad()
def efficient_kan_activations(model: EfficientKAN, x: torch.Tensor, device):
    activations = []
    removable_hooks = []

    x = x.to(device)
    model = model.to(device)

    activations.append(x)

    # Register Hooks
    for kan_layer in model.layers:
        kan_layer: EfficientKANLinear

        def hook_track_activation(module, input, output):
            activations.append(output)

        removable_hooks.append(kan_layer.register_forward_hook(hook_track_activation))

    # Forward Pass
    model(x)

    # Remove All Hooks
    for _hook in removable_hooks:
        _hook.remove()

    return activations


@torch.no_grad()
def pnkan_activations(model: WiseNet, x: torch.Tensor, task_id: int, device):
    activations = []
    removable_hooks = []

    x = x.to(device)
    model = model.to(device)

    activations.append(x)

    # Register Hooks
    for kan_layer in model.layers:
        kan_layer: PnKANLinear

        def hook_track_activation(module, input, output):
            activations.append(output)

        removable_hooks.append(kan_layer.register_forward_hook(hook_track_activation))

    # Forward Pass
    model(x, task_id)

    # Remove All Hooks
    for _hook in removable_hooks:
        _hook.remove()

    return activations
