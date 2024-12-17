import logging
import os
from functools import partial
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
from claiutil.torch.preload import preload
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import clkan.config as cfg
import wandb
from clkan.model import AboutModel
from clkan.regression.base import LitRegression
from clkan.regression.ewc import EWC
from clkan.regression.packnet import LitPackNet
from clkan.regression.si import SI
from clkan.regression.wisekan import LitWiseKAN
from clkan.regression.wisemlp import LitWiseMLP
from clkan.scenario import AboutScenario, Scenario

logger = logging.getLogger(__name__)

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


@torch.no_grad()
def plot_figure_1D_dataset(
    model: L.LightningModule,
    logger: L.logging.Logger,
    task_id: int,
    truth_dataset: Dataset,
    device,
):
    model = model.eval().to(device)
    resolution = 1000
    x = torch.linspace(-1, 1, resolution, device=device).view(-1, 1)
    y: Tensor = model(x)

    # Randomly sample num_truth points from the truth dataset
    indices = torch.randperm(len(truth_dataset))[:resolution]
    x_truth = torch.stack(
        [truth_dataset[i][0] for i in indices]
    )  # Shape (num_truth, 1)
    y_truth = torch.stack(
        [truth_dataset[i][1] for i in indices]
    )  # Shape (num_truth, 1)
    # Sort the truth dataset by x so we can plot it
    x_truth, indices = x_truth.sort(0)
    y_truth = y_truth[indices]

    with plt.ioff():
        fig, ax = plt.subplots()
        ax.plot(x.cpu(), y.cpu(), label="Model")
        ax.plot(x_truth.flatten(), y_truth.flatten(), label="Truth", linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(-1, 1)
        ax.set_title(f"After Completing Task {task_id}")
        ax.legend()
        logger.experiment.log({"1D_figure": wandb.Image(fig)})


def lighting_cl_loop(
    config: cfg.Config,
    model: nn.Module,
    about_model: AboutModel,
    scenario: Scenario,
    about_scenario: AboutScenario,
    work_dir: Path,
    experiment_name: str,
):
    # preload datasets
    if config.model.type_ == "WiseKAN":
        logger.info("Using WiseKAN model")
        model = LitWiseKAN(config, about_scenario, scenario, about_model, model)
    elif config.model.type_ == "PackNet":
        logger.info("Using PackNet model")
        model = LitPackNet(config, about_scenario, scenario, about_model, model)
    elif config.model.type_ == "WiseMLP":
        logger.info("Using WiseMLP model")
        model = LitWiseMLP(config, about_scenario, scenario, about_model, model)
    elif config.strategy and config.strategy.type_ == "EWC":
        logger.info("Using EWC strategy")
        model = EWC(config, about_scenario, scenario, about_model, model)
    elif config.strategy and config.strategy.type_ == "SI":
        logger.info("Using SI strategy")
        model = SI(config, about_scenario, scenario, about_model, model)
    else:
        model = LitRegression(config, about_scenario, scenario, about_model, model)
    assert isinstance(model, L.LightningModule)
    model = model.to(config.training.device)

    for task_id, (train_ds, test_ds, valid_ds) in enumerate(
        zip(scenario.train_stream, scenario.test_stream, scenario.valid_stream)
    ):
        logger.info(
            f"task {task_id}: train(n={len(train_ds)}), test(n={len(test_ds)}), valid(n={len(valid_ds)})"
        )

    project_name = "clkan"
    log_dir = Path(os.environ.get("WANDB_LOG_ROOT", "./wandb")) / project_name
    log_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = WandbLogger(
        project=project_name, save_dir=log_dir, name=experiment_name, tags=config.tags
    )
    if config.watch_gradients:
        metric_logger.watch(model)

    # Add tags if they are present
    tags = []
    if config.tags is not None:
        tags.extend(config.tags)
    metric_logger.experiment.tags = tags

    trainer = L.Trainer(
        max_epochs=config.training.initial_task_epochs,
        default_root_dir=work_dir,
        logger=metric_logger,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        deterministic=False,
        # devices=[int(config.training.device.split(":")[-1])],
    )
    model.task_epochs = config.training.initial_task_epochs

    joint_test_dataset = ConcatDataset(scenario.test_stream)
    new_train_loader = partial(
        DataLoader,
        batch_size=config.training.train_mb_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    new_test_loader = partial(
        DataLoader,
        batch_size=config.training.eval_mb_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    # Since we are using lightning for continual learning we need to manually
    # set the global step. This is the number of batches seen by the model.
    # By default lightning starts at 0 for each new fit() call. We want to
    # keep the global step across all tasks.
    global_step_override = 0

    if config.training.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    metrics = dict()
    for task_id, train in enumerate(scenario.train_stream):
        logger.info(f"Task {task_id + 1}/{about_scenario.num_tasks}")
        model.train_task_id = task_id
        preloaded = preload(train, 0)
        train_loader = new_train_loader(preloaded, collate_fn=preloaded.collate_fn)
        # train_loader = new_train_loader(train)
        valid_loader = (
            new_test_loader(scenario.valid_stream[task_id])
            if scenario.valid_stream is not None
            else None
        )
        logger.info(f"Preloaded task {task_id} dataset")

        trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_override
        trainer.fit(model, train_loader, val_dataloaders=valid_loader)
        # The number of epochs per task is added to the total number of epochs
        model._task_epoch_offset = trainer.fit_loop.max_epochs
        trainer.fit_loop.max_epochs += model.config.training.epochs
        model.task_epochs = model.config.training.epochs
        global_step_override = trainer.global_step

        # If pytorch lightning was interrupted we can quit
        if trainer.interrupted:
            break

        # Eval Loop ------------------------------------------------------------
        for test_task_id, test in enumerate(scenario.test_stream):
            model.test_task_id = test_task_id
            test_loader = new_test_loader(test)

            metrics = trainer.test(model, test_loader, verbose=False)
        model.on_test_tasks_end()

        trainer.optimizers = [model.configure_optimizers()]

        if about_scenario.in_features == 1:
            plot_figure_1D_dataset(
                model,
                metric_logger,
                task_id,
                joint_test_dataset,
                config.training.device,
            )

    model.on_stream_end()

    if config.training.device == "cuda":
        metric_logger.log_metrics(
            {"Peak GPU Memory (Bytes)": torch.cuda.max_memory_allocated()}
        )

    wandb.finish(0, True)

    # Negative because we want to maximize the r2 score and the optimizer minimizes
    match (model.joint, config.target_metric):
        case (True, "R2"):
            return -metrics[-1].get("test_task/00_r2"), model.count_parameters()
        case (True, "RMSE"):
            return metrics[-1].get("test_task/00_rmse"), model.count_parameters()
        case (False, "R2"):
            return -model.cl_r2.scenario_average(), model.count_parameters()
        case (False, "RMSE"):
            return model.cl_rmse.scenario_average(), model.count_parameters()
