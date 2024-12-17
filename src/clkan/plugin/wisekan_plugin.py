import logging
from copy import deepcopy
from typing import Callable, Optional

import torch
from clkan import config
from clkan.model.wisekan import WiseNet
from clkan.prune import AGPSchedule, EarlyStopping
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import Metric

logger = logging.getLogger(__name__)


class WiseKANPlugin:
    def __init__(
        self,
        start_edge_prune_percent: float | None,
        start_coef_prune_percent: float,
        min_sparsity: float,
        early_stopping_threshold: float,
        target_sparsity: float = 1.0,
    ):
        self.start_edge_prune_percent = start_edge_prune_percent
        self.start_coef_prune_percent = start_coef_prune_percent

        self.agp_edge = AGPSchedule(0, target_sparsity)
        self.agp_unit = AGPSchedule(0, target_sparsity)
        self.is_pruning_edge = (
            True if self.start_edge_prune_percent is not None else False
        )
        self.is_pruning_coef = True

        self.early_stopping = EarlyStopping(min_sparsity, early_stopping_threshold)

        self.model_state = None

        self.loader: Optional[DataLoader] = None

    def is_pruning(self, step: int) -> bool:
        edge_pruning = self.is_pruning_edge and self.agp_edge.has_started(step)
        coef_pruning = self.is_pruning_coef and self.agp_unit.has_started(step)
        return edge_pruning or coef_pruning

    @staticmethod
    def from_wisekan(config: config.WiseKAN):
        return WiseKANPlugin(
            start_edge_prune_percent=config.start_edge_prune_percent,
            start_coef_prune_percent=config.start_coef_prune_percent,
            min_sparsity=config.min_sparsity,
            early_stopping_threshold=config.early_stopping_threshold,
            target_sparsity=config.target_sparsity,
        )

    def _setup(self, train_epochs):
        self.agp_edge.total_steps = train_epochs
        self.agp_unit.total_steps = train_epochs

        if self.start_edge_prune_percent is not None:
            self.agp_edge.start_step = int(train_epochs * self.start_edge_prune_percent)
        self.agp_unit.start_step = int(train_epochs * self.start_coef_prune_percent)

        logger.info(
            f"Start AGP Edge Prune @ {self.agp_edge.start_step}, Start AGP Coef Prune @ {self.agp_unit.start_step}"
        )

    def _before_training_exp(self, data_loader: DataLoader):
        self.is_pruning_edge = (
            True if self.start_edge_prune_percent is not None else False
        )
        self.is_pruning_coef = True
        self.loader = data_loader
        self.early_stopping.best_score = None

    def _before_training_epoch(
        self,
        step: int,
        device: torch.device,
        task_id: int,
        model: WiseNet,
        spline_importance_override: Optional[Tensor] = None,
        coef_importance_override: Optional[Tensor] = None,
    ):
        isinstance(model, nn.Module)
        # Save model state for rollback
        self.model_state = deepcopy(model.state_dict())

        edge_sparsity = 0.0
        coef_sparsity = 0.0
        if self.is_pruning(step):
            self.old_sparsity = model.coef_task_sparsity(task_id)

            if self.is_pruning_edge and self.agp_edge.has_started(step):
                edge_sparsity = self.agp_edge(step)
                model.spline_prune(task_id, edge_sparsity, spline_importance_override)

            if self.is_pruning_coef and self.agp_unit.has_started(step):
                coef_sparsity = self.agp_unit(step)
                model.coef_prune(task_id, coef_sparsity, coef_importance_override)

        logger.info(
            "Target Sparsity: edge={:.2f}, coef={:.2f}".format(
                edge_sparsity, coef_sparsity
            )
        )

    def _after_training_epoch(
        self,
        step: int,
        task_id: int,
        model: WiseNet,
        metric: Metric,
        device: torch.device,
        log_callback: Callable[[str, float], None],
    ):
        assert isinstance(model, nn.Module)
        assert isinstance(model, WiseNet)

        # Perform validation to decide if pruning should stop.
        score = self.calculate_validation_accuracy(model, metric, task_id, device)

        if self.is_pruning(step):
            # If early stopping is triggered we rollback the model. And disable a
            # pruning type.
            stop = self.early_stopping(score, self.old_sparsity)
            if stop and self.is_pruning_edge:
                logger.info("Edge pruning early stopping triggered. Rolling back model")
                model.load_state_dict(self.model_state)
                self.is_pruning_edge = False
            elif stop and self.is_pruning_coef:
                logger.info("Coef pruning early stopping triggered. Rolling back model")
                model.load_state_dict(self.model_state)
                self.is_pruning_coef = False

        coef_sparsity = model.coef_task_sparsity(task_id)
        edge_sparsity = model.spline_task_sparsity(task_id)
        global_sparsity = model.coef_global_sparsity(task_id + 1)
        # I have never used this python syntax before.
        score and log_callback("wisekan/score", score)
        edge_sparsity and log_callback("wisekan/edge_sparsity", edge_sparsity)
        coef_sparsity and log_callback("wisekan/coef_sparsity", coef_sparsity)
        global_sparsity and log_callback("wisekan/global_sparsity", global_sparsity)

    def calculate_validation_accuracy(
        self,
        model: WiseNet,
        metric: Metric,
        task_id: int,
        device,
    ) -> float:
        with torch.no_grad():
            metric.reset()
            metric = metric.to(device)

            mode = model.training
            model.eval()

            for x, y in self.loader:
                x: torch.Tensor = x.to(device)
                y: torch.Tensor = y.to(device)
                y_pred = model.forward(x, task_id)
                metric.update(y_pred, y)

            model.train(mode)
        return float(metric.compute().item())
