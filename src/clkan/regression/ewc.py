import logging

import clkan.config as cfg
from claiutil.regularization import (
    copy_parameters,
    ewc,
    parameters_like_module,
    quadratic_penalty,
)
from clkan.model import AboutModel
from clkan.scenario import AboutScenario, Scenario
from torch import nn
from torch.utils.data import DataLoader

from .base import LitRegression

logger = logging.getLogger(__name__)


class EWC(LitRegression):
    def __init__(
        self,
        config: cfg.Config,
        about_scenario: AboutScenario,
        scenario: Scenario,
        about_model: AboutModel,
        model: nn.Module,
    ):
        ewc_config = config.strategy
        if not isinstance(ewc_config, cfg.EWC):
            raise ValueError("EWC strategy must be used with EWC model")
        super().__init__(config, about_scenario, scenario, about_model, model)

        self.importances = parameters_like_module(model)
        self.pre_task_model = parameters_like_module(model)
        self.ewc_lambda = ewc_config.ewc_lambda

    def on_train_start(self) -> None:
        copy_parameters(self.pre_task_model, self.model.parameters())

    def training_step(self, batch, batch_idx) -> nn.Module:
        loss = super().training_step(batch, batch_idx)
        ewc_loss = self.ewc_lambda * quadratic_penalty(
            self.model.parameters(),
            self.pre_task_model,
            self.importances,
        )
        self.log("train/ewc", ewc_loss, on_epoch=True, on_step=True)
        return loss + ewc_loss

    def on_train_end(self) -> None:
        logger.info("Computing Fisher importance. This loops over the training data.")
        dataloader = DataLoader(
            self.scenario.train_stream[self.train_task_id],
            batch_size=self.config.training.eval_mb_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
        )
        self.importances = ewc.ewc_fisher_update_importances(
            self.importances, self.model, self.loss_func, dataloader, self.device
        )
