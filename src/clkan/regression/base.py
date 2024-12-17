import logging
import math
from dataclasses import asdict
from typing import Any

import clkan.config as cfg
import lightning as L
import torch
from claiutil.metrics import CLMetric
from clkan.model import AboutModel
from clkan.scenario import AboutScenario, Scenario
from torch import Tensor, nn
from torch.optim import Adam
from torchmetrics import MeanSquaredError, R2Score

import wandb

logger = logging.getLogger(__name__)


class LitRegression(L.LightningModule):
    train_task_id: int
    """The task id of the current training phase"""
    test_task_id: int
    """The task id of the current testing phase"""
    _task_epoch_offset: int = 0
    """The epoch offset of the current task"""
    task_epochs: int = 0
    """The number of epochs for the current task"""

    def __init__(
        self,
        config: cfg.Config,
        about_scenario: AboutScenario,
        scenario: Scenario,
        about_model: AboutModel,
        model: nn.Module,
    ):
        super().__init__()
        self.scenario = scenario
        self.model = model
        self.config = config
        self.about_scenario = about_scenario
        self.about_model = about_model
        self.joint = about_scenario.num_tasks == 1

        self.save_hyperparameters(
            {"config": config.model_dump(), "model": asdict(about_model)}
        )
        self.loss_func = lambda y_hat, y: (y_hat - y).square().mean(dim=0).sum()
        # Metrics
        self.r2 = R2Score(multioutput="raw_values")
        self.mse = MeanSquaredError(num_outputs=int(about_scenario.out_features))
        self.cl_r2 = CLMetric(about_scenario.num_tasks)
        self.cl_mse = CLMetric(about_scenario.num_tasks)
        self.cl_rmse = CLMetric(about_scenario.num_tasks)
        self.task_mask = nn.Parameter(about_scenario.task_mask, requires_grad=False)

    @property
    def current_task_epoch(self) -> int:
        return self.current_epoch - self._task_epoch_offset

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def train_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def test_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def val_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)

    # Training Loop ------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        self.reset_metrics()

    @torch.no_grad()
    def train_task_mask(self, y: Tensor) -> Tensor:
        # HACK: When the targets are orthogonalized, and the model is learning
        # the joint problem, we need to mask the targets of the ``current" task
        # to avoid the model wasting capacity on learning the zero targets.
        # This is a hack because we don't use the task mask but rather rely on
        # the fact that some of the targets are zero i.e. the mask is implicit.
        if self.joint and self.config.scenario.orthogonalize_targets:
            return y != 0
        return self.task_mask[self.train_task_id]

    @torch.no_grad()
    def test_task_mask(self, y: Tensor) -> Tensor:
        # See the comment in ``train_task_mask"
        if self.joint and self.config.scenario.orthogonalize_targets:
            return y != 0
        return self.task_mask[self.test_task_id]

    def training_step(self, batch, batch_idx) -> Tensor:
        x: Tensor = batch[0]
        y: Tensor = batch[1]

        y_hat = self.train_task_mask(y) * self.train_forward(x)
        loss = self.loss_func(y_hat, y)

        self.mse.update(y_hat, y)
        self.r2.update(y_hat, y)
        self.log("train/loss", loss, on_epoch=True, on_step=True)
        return loss

    def on_train_epoch_end(self) -> None:
        mask = self.task_mask[self.train_task_id]
        mse = (
            self.mse.compute()[mask].mean().item()
            if self.mse.num_outputs != 1
            else self.mse.compute().item()
        )
        r2 = self.r2.compute()[mask].mean().item()

        wandb.log({"train/mse": mse, "train/r2": r2})

        logger.info(
            f"Epoch {self.current_task_epoch}/{self.task_epochs}: R2 {r2:.4}; RMSE: {math.sqrt(mse):.2e}"
        )

    # Validation Loop ----------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self.reset_metrics()

    def validation_step(self, batch, batch_idx) -> Tensor:
        x: Tensor = batch[0]
        y: Tensor = batch[1]
        y_hat = self.train_task_mask(y) * self.val_forward(x)
        loss = self.loss_func(y_hat, y)

        self.r2.update(y_hat, y)
        self.mse.update(y_hat, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        mask = self.task_mask[self.train_task_id]

        mse = (
            self.mse.compute()[mask].mean().item()
            if self.mse.num_outputs != 1
            else self.mse.compute().item()
        )
        r2 = self.r2.compute()[mask].mean().item()
        wandb.log({"val/mse": mse, "val/r2": r2, "val/rmse": math.sqrt(mse)})

    # Testing Loop -------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self.reset_metrics()

    def test_step(self, batch, batch_idx) -> Tensor:
        x: Tensor = batch[0]
        y: Tensor = batch[1]
        y_hat = self.test_task_mask(y) * self.test_forward(x)
        loss = self.loss_func(y_hat, y)

        self.r2.update(y_hat, y)
        self.mse.update(y_hat, y)
        return loss

    def on_test_epoch_end(self) -> None:
        mask = self.task_mask[self.test_task_id]
        r2 = self.r2.compute()[mask].mean().item()
        mse = (
            self.mse.compute()[mask].mean().item()
            if self.mse.num_outputs != 1
            else self.mse.compute().item()
        )
        rmse = math.sqrt(mse)

        self.cl_mse.add(self.train_task_id, self.test_task_id, mse)
        self.cl_r2.add(self.train_task_id, self.test_task_id, r2)
        self.cl_rmse.add(self.train_task_id, self.test_task_id, rmse)

        self.log(f"test_task/{self.test_task_id:02d}_mse", mse, on_epoch=True)
        self.log(f"test_task/{self.test_task_id:02d}_r2", r2, on_epoch=True)
        self.log(f"test_task/{self.test_task_id:02d}_rmse", rmse, on_epoch=True)

    def on_test_tasks_end(self) -> None:
        if not self.joint:
            metrics = {}
            metrics["test/seen_mse"] = self.cl_mse.seen_task_average(self.train_task_id)
            metrics["test/seen_r2"] = self.cl_r2.seen_task_average(self.train_task_id)
            metrics["parameter_count"] = self.count_parameters()
            metrics["test/plasticity"] = (
                self.cl_r2._results.diag()[: self.train_task_id + 1].mean().item()
            )
            wandb.log(metrics)

    def on_stream_end(self) -> None:
        if not self.joint:
            metrics = {}
            metrics["stream/AVG_R2"] = self.cl_r2.scenario_average()
            metrics["stream/FWT_R2"] = self.cl_r2.forward_transfer()
            metrics["stream/BWT_R2"] = self.cl_r2.backward_transfer()
            metrics["stream/AVG_RMSE"] = self.cl_rmse.scenario_average()
            metrics["stream/AVG_MSE"] = self.cl_mse.scenario_average()
            metrics["stream/FWT_MSE"] = self.cl_mse.forward_transfer()
            metrics["stream/BWT_MSE"] = self.cl_mse.backward_transfer()
            wandb.log(metrics)

    def configure_optimizers(self):
        if self.config.training.optimizer == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.lr,
            )
        elif self.config.training.optimizer == "Adam":
            return Adam(self.model.parameters(), lr=self.config.training.lr)
        elif self.config.training.optimizer == "LBFGS":
            return torch.optim.LBFGS(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.training.lr,
            )
        else:
            raise ValueError("Invalid optimizer")

    def reset_metrics(self):
        self.mse.reset()
        self.r2.reset()

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
