import logging

import clkan.config as cfg
from clkan.model import AboutModel
from clkan.model.wisemlp import PackNet
from clkan.plugin.wisekan_plugin import WiseKANPlugin
from clkan.regression.base import LitRegression
from clkan.scenario import AboutScenario, Scenario
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score

logger = logging.getLogger(__name__)


class LitWiseMLP(LitRegression):
    model: PackNet

    def __init__(
        self,
        config: cfg.Config,
        about_scenario: AboutScenario,
        scenario: Scenario,
        about_model: AboutModel,
        model: nn.Module,
    ):
        super().__init__(config, about_scenario, scenario, about_model, model)
        self.wise_mlp_cfg = config.model
        self.valid_stream = scenario.valid_stream
        self.prune_metric = R2Score(
            num_outputs=int(about_scenario.out_features),
            multioutput="variance_weighted",
        )

        assert isinstance(self.wise_mlp_cfg, cfg.WiseMLP)
        assert scenario.valid_stream is not None

        self.wise_kan_plugin = WiseKANPlugin(
            start_edge_prune_percent=None,
            start_coef_prune_percent=self.wise_mlp_cfg.start_coef_prune_percent,
            min_sparsity=self.wise_mlp_cfg.min_sparsity,
            early_stopping_threshold=self.wise_mlp_cfg.early_stopping_threshold,
            target_sparsity=self.wise_mlp_cfg.target_sparsity,
        )

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

        self.wise_kan_plugin._setup(self.config.training.epochs)
        assert self.config.training.initial_task_epochs == self.config.training.epochs

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
