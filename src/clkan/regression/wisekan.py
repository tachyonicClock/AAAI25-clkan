import logging

import clkan.config as cfg
from clkan.model import AboutModel
from clkan.model.wisekan import WiseKAN
from clkan.plugin.wisekan_plugin import WiseKANPlugin
from clkan.regression.base import LitRegression
from clkan.scenario import AboutScenario, Scenario
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score

logger = logging.getLogger(__name__)


class LitWiseKAN(LitRegression):
    model: WiseKAN

    def __init__(
        self,
        config: cfg.Config,
        about_scenario: AboutScenario,
        scenario: Scenario,
        about_model: AboutModel,
        model: nn.Module,
    ):
        super().__init__(config, about_scenario, scenario, about_model, model)
        assert isinstance(config.model, cfg.WiseKAN)
        self.wise_kan_plugin = WiseKANPlugin.from_wisekan(config.model)
        assert scenario.valid_stream is not None
        self.valid_stream = scenario.valid_stream

        self.prune_metric = R2Score(multioutput="variance_weighted")

    def training_step(self, batch, batch_idx) -> Tensor:
        loss = super().training_step(batch, batch_idx)

        kan_cfg: cfg.EfficientKAN = self.config.model
        reg_loss = self.model.regularization_loss(
            self.train_task_id,
            kan_cfg.activation_penalty_weight,
            kan_cfg.entropy_penalty_weight,
        )
        self.log("train/regularization", reg_loss, on_epoch=True, on_step=True)
        return loss + reg_loss

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        return self.model.forward(x, task_id)

    def train_forward(self, x: Tensor) -> Tensor:
        return self.forward(x, self.train_task_id)

    def test_forward(self, x: Tensor) -> Tensor:
        return self.forward(x, self.test_task_id)

    def val_forward(self, x: Tensor) -> Tensor:
        return self.forward(x, self.train_task_id)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        agp_edge, agp_unit = (
            self.wise_kan_plugin.agp_edge,
            self.wise_kan_plugin.agp_unit,
        )
        agp_edge.total_steps = self.task_epochs
        agp_unit.total_steps = self.task_epochs

        agp_edge.start_step = int(
            self.config.training.epochs * self.wise_kan_plugin.start_edge_prune_percent
        )
        agp_unit.start_step = int(
            self.config.training.epochs * self.wise_kan_plugin.start_coef_prune_percent
        )

        # Adjust start step for pruning in the first task because of the warmup epochs
        agp_edge.start_step += self.task_epochs - self.config.training.epochs
        agp_unit.start_step += self.task_epochs - self.config.training.epochs

        logger.info(
            f"Start AGP Edge Prune @ {agp_edge.start_step}, Start AGP Coef Prune @ {agp_unit.start_step}"
        )

        self.wise_kan_plugin._before_training_exp(
            DataLoader(
                self.valid_stream[self.train_task_id],
                batch_size=self.config.training.eval_mb_size,
                shuffle=False,
                num_workers=self.config.training.num_workers,
            )
        )

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.wise_kan_plugin._before_training_epoch(
            self.current_task_epoch,
            self.device,
            self.train_task_id,
            self.model,
        )

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.wise_kan_plugin._after_training_epoch(
            self.current_task_epoch,
            self.train_task_id,
            self.model,
            self.prune_metric,
            self.device,
            self.log,
        )

    def count_parameters(self):
        return self.model.count_parameters(self.train_task_id)
