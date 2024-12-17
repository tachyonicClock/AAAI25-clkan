from dataclasses import dataclass
from typing import Tuple, Type

from efficient_kan import KAN as EfficientKAN  # type: ignore
from torch import Tensor, nn

import clkan.config as cfg
from clkan.model.mlp import MLP
from clkan.model.wisekan import WiseKAN
from clkan.model.wisemlp import PackNet
from clkan.scenario import AboutScenario


@dataclass
class AboutModel:
    """This object contains metadata about the model that is used to setup other
    parts of the training process. This metadata cannot be included in the
    configuration because it is not known until the model is built and is more
    conveniently inferred from the model itself.
    """

    trainable_parameters: int = 0


def activation_function(activation: cfg.ActivationEnum) -> Type[nn.Module]:
    def _zero(x: Tensor) -> Tensor:
        return x.zero_()

    return {
        cfg.ActivationEnum.ReLU: nn.ReLU,
        cfg.ActivationEnum.Zero: lambda: _zero,
        cfg.ActivationEnum.SiLU: nn.SiLU,
        cfg.ActivationEnum.Sigmoid: nn.Sigmoid,
        cfg.ActivationEnum.Identity: nn.Identity,
    }[activation]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def new_model(
    config: cfg.ModelTypes, about_scenario: AboutScenario, device
) -> Tuple[AboutModel, nn.Module]:
    in_features = about_scenario.in_features
    out_features = about_scenario.out_features

    match config.type_:
        case "MLP":
            isinstance(config, cfg.MLP)
            if config.n_hidden_layers is not None:
                layers_hidden = config.layers_hidden[: config.n_hidden_layers]
            else:
                layers_hidden = config.layers_hidden

            layer_features = [
                about_scenario.in_features,
                *layers_hidden,
                about_scenario.out_features,
            ]
            model = MLP(layer_features, activation_function(config.activation)())
        case "EfficientKAN":
            assert isinstance(config, cfg.EfficientKAN)
            if config.n_hidden_layers is not None:
                layers_hidden = config.layers_hidden[: config.n_hidden_layers]
            else:
                layers_hidden = config.layers_hidden

            model = EfficientKAN(
                layers_hidden=[in_features] + layers_hidden + [out_features],
                grid_size=config.grid_size,
                spline_order=config.spline_order,
                scale_noise=config.scale_noise,
                scale_base=config.scale_base,
                scale_spline=config.scale_spline,
                base_activation=activation_function(config.base_activation),
                grid_eps=config.grid_eps,
                grid_range=config.grid_range,
                enable_base_weight=config.enable_base_weight,
                enable_standalone_scale_spline=config.enable_standalone_scale_spline,
                norm=config.norm,
                first_layer_is_linear=config.first_layer_is_linear,
                last_layer_is_linear=config.last_layer_is_linear,
            )
        case "WiseKAN":
            assert isinstance(config, cfg.WiseKAN)
            assert about_scenario.task_mask is not None
            if config.n_hidden_layers is not None:
                layers_hidden = config.layers_hidden[: config.n_hidden_layers]
            else:
                layers_hidden = config.layers_hidden

            model = WiseKAN(
                layers_hidden=[in_features] + layers_hidden + [out_features],
                task_masks=about_scenario.task_mask,
                grid_size=config.grid_size,
                spline_order=config.spline_order,
                scale_noise=config.scale_noise,
                scale_base=config.scale_base,
                scale_spline=config.scale_spline,
                base_activation=activation_function(config.base_activation),
                grid_eps=config.grid_eps,
                grid_range=config.grid_range,
                enable_base_weight=config.enable_base_weight,
                enable_standalone_scale_spline=config.enable_standalone_scale_spline,
                first_layer_is_linear=config.first_layer_is_linear,
                last_layer_is_linear=config.last_layer_is_linear,
                norm=config.norm,
            )
        case "WiseMLP" | "PackNet":
            assert isinstance(config, (cfg.WiseMLP, cfg.PackNet))
            assert about_scenario.task_mask is not None
            if config.n_hidden_layers is not None:
                layers_hidden = config.layers_hidden[: config.n_hidden_layers]
            else:
                layers_hidden = config.layers_hidden

            layer_features = [
                about_scenario.in_features,
                *layers_hidden,
                about_scenario.out_features,
            ]
            model = PackNet(layer_features, about_scenario.task_mask)

        case _:
            raise ValueError(f"Unknown model type: {config.type_}")
    return AboutModel(trainable_parameters=count_trainable_parameters(model)), model
