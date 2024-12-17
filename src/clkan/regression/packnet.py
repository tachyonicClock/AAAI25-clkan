import logging

import clkan.config as cfg
from clkan.model import AboutModel
from clkan.model.wisemlp import PackNet
from clkan.scenario import AboutScenario, Scenario
from torch import Tensor, nn

from .base import LitRegression

logger = logging.getLogger(__name__)


class LitPackNet(LitRegression):
    model: PackNet

    def __init__(
        self,
        config: cfg.Config,
        about_scenario: AboutScenario,
        scenario: Scenario,
        about_model: AboutModel,
        model: nn.Module,
    ):
        assert isinstance(config.model, cfg.PackNet)
        assert isinstance(model, PackNet)
        self.config_model = config.model
        super().__init__(config, about_scenario, scenario, about_model, model)

        self.start_epoch = int(
            config.training.epochs * self.config_model.prune_after_p_percent
        )
        logger.info(f"Pruning will start at epoch {self.start_epoch}")

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        return self.model.forward(x, task_id)

    def train_forward(self, x: Tensor) -> Tensor:
        return self.forward(x, self.train_task_id)

    def test_forward(self, x: Tensor) -> Tensor:
        return self.forward(x, self.test_task_id)

    def val_forward(self, x: Tensor) -> Tensor:
        return self.forward(x, self.train_task_id)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.current_task_epoch == self.start_epoch:
            logger.info(f"Pruning Model {self.config_model.prune_ratio}")
            self.model.coef_prune(self.train_task_id, self.config_model.prune_ratio)
            self.log(
                "packnet/global_sparsity",
                self.model.coef_global_sparsity(self.train_task_id + 1),
            )

    def count_parameters(self):
        return self.model.count_parameters(self.train_task_id)
