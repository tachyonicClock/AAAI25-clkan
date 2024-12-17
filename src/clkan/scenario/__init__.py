import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import torch
from claiutil.datasets import (
    get_targets,
)
from numpy import prod
from torch.utils.data import Dataset, Subset

import clkan.config as cfg
from clkan.scenario.gaussian_peaks import gaussian_peaks_dataset
from clkan.scenario.incremental_feynman import load_incremental_feynman
from clkan.scenario.task_incremental_regression import task_incremental_regression

logger = logging.getLogger(__name__)


def validation_split(
    dataset: Dataset, valid_proportion: float, add_targets: bool = True
) -> Tuple[Dataset, Dataset]:
    valid_size = int(len(dataset) * valid_proportion)
    train_size = len(dataset) - valid_size
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    if add_targets:
        valid_set.targets = get_targets(dataset)[valid_indices]
        train_set.targets = get_targets(dataset)[train_indices]
    return train_set, valid_set


def validation_split_streams(
    stream: Sequence[Dataset], valid_proportion: float
) -> Tuple[Sequence[Dataset], Optional[Sequence[Dataset]]]:
    if valid_proportion <= 0:
        return stream, None

    train_stream = []
    valid_stream = []
    for dataset in stream:
        train_set, valid_set = validation_split(
            dataset, valid_proportion, add_targets=False
        )
        train_stream.append(train_set)
        valid_stream.append(valid_set)
    return train_stream, valid_stream


def get_torch_data_dir():
    torch_data_dir = os.environ.get("TORCH_DATA_DIR", None)
    if torch_data_dir is None:
        raise ValueError("Environment variable TORCH_DATA_DIR is not set")
    return torch_data_dir


@dataclass(frozen=True)
class AboutScenario:
    in_features: int
    out_features: int
    num_tasks: int
    is_regression: bool
    class_schedule: Optional[Sequence[Set[int]]]
    task_mask: Optional[torch.Tensor]

    @property
    def class_order(self):
        # Return a flat list of classes in the order they appear in the class
        # schedule
        return [c for task in self.class_schedule for c in task]


@dataclass()
class Scenario:
    train_stream: List[Dataset]
    test_stream: List[Dataset]
    valid_stream: Optional[List[Dataset]]


def new_scenario(config: cfg.ScenarioTypes) -> Tuple[AboutScenario, Scenario]:
    if isinstance(config, cfg.GaussianPeaks):
        train_stream, test_stream = gaussian_peaks_dataset(config)
        return AboutScenario(
            in_features=1,
            out_features=1,
            is_regression=True,
            num_tasks=5,
            class_schedule=None,
        ), Scenario(train_stream, test_stream, None)
    elif isinstance(config, cfg.TaskIncrementalRegression):
        train_stream, test_stream, out_mask = task_incremental_regression(
            get_torch_data_dir(),
            config.hdf5_filename,
            config.num_tasks,
            config.test_proportion,
            config.orthogonalize_inputs,
            config.orthogonalize_targets,
        )

        valid_stream = None
        if config.valid_proportion > 0:
            train_stream, valid_stream = validation_split_streams(
                train_stream, config.valid_proportion
            )

        in_features = prod(train_stream[0][0][0].shape)
        out_features = prod(train_stream[0][0][1].shape)
        return AboutScenario(
            in_features=int(in_features),
            out_features=int(out_features),
            num_tasks=config.num_tasks,
            is_regression=True,
            class_schedule=None,
            task_mask=out_mask,
        ), Scenario(train_stream, test_stream, valid_stream)
    elif isinstance(config, cfg.IncrementalFeynman):
        train_stream, test_stream, task_mask = load_incremental_feynman(
            get_torch_data_dir(),
            config.hdf5_filename,
            config.shuffle,
            config.joint,
            config.standard_score_normalize,
            config.min_max_normalize,
        )

        valid_stream = None
        if config.valid_proportion > 0:
            train_stream, valid_stream = validation_split_streams(
                train_stream, config.valid_proportion
            )

        return AboutScenario(
            in_features=83,
            out_features=34,
            num_tasks=len(train_stream),
            is_regression=True,
            class_schedule=None,
            task_mask=task_mask,
        ), Scenario(train_stream, test_stream, valid_stream)
    else:
        raise ValueError(
            f"Building Scenario failed because it was given an unknown scenario"
            f" type: {config}"
        )


__all__ = ["AboutScenario", "new_scenario"]
