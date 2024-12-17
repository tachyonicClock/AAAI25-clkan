import clkan.config as cfg
from claiutil.regularization import (
    copy_parameters,
    parameters_like_module,
    quadratic_penalty,
    si,
    zero_parameters,
)
from clkan.model import AboutModel
from clkan.scenario import AboutScenario, Scenario
from torch import nn

from .base import LitRegression


class SI(LitRegression):
    def __init__(
        self,
        config: cfg.Config,
        about_scenario: AboutScenario,
        scenario: Scenario,
        about_model: AboutModel,
        model: nn.Module,
    ):
        si_config = config.strategy
        if not isinstance(si_config, cfg.SI):
            raise ValueError("EWC strategy must be used with EWC model")
        super().__init__(config, about_scenario, scenario, about_model, model)

        self.task_trajectory = parameters_like_module(model)
        self.importances = parameters_like_module(model)
        self.pre_task_parameters = parameters_like_module(model)
        self.pre_step_parameters = parameters_like_module(model)
        self.si_lambda = si_config.si_lambda
        self.epsilon = si_config.epsilon

    def on_train_start(self) -> None:
        zero_parameters(self.task_trajectory)
        copy_parameters(self.pre_task_parameters, self.model.parameters())

    def training_step(self, batch, batch_idx) -> nn.Module:
        copy_parameters(self.pre_step_parameters, self.model.parameters())
        loss = super().training_step(batch, batch_idx)

        si_loss = self.si_lambda * quadratic_penalty(
            self.model.parameters(), self.pre_task_parameters, self.importances
        )

        self.log("train/si", si_loss, on_epoch=True, on_step=True)

        return loss + si_loss

    def on_after_optimizer_step(self) -> None:
        self.task_trajectory = si.si_update_trajectory(
            self.task_trajectory, self.model.parameters(), self.pre_step_parameters
        )

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.on_after_optimizer_step()

    def on_train_end(self) -> None:
        self.importances = si.si_update_importances(
            self.importances,
            self.model.parameters(),
            self.pre_task_parameters,
            self.task_trajectory,
            self.epsilon,
        )
