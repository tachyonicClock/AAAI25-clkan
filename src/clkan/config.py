from enum import Enum
from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class ConfigBase(BaseModel):
    # Forbid extra fields
    model_config = ConfigDict(extra="forbid")


class ActivationEnum(str, Enum):
    SiLU: str = "SiLU"
    ReLU: str = "ReLU"
    Identity: str = "Identity"
    Sigmoid: str = "Sigmoid"
    Zero: str = "Zero"


# Model ------------------------------------------------------------------------


class EfficientKAN(ConfigBase):
    type_: Literal["EfficientKAN"] = "EfficientKAN"
    layers_hidden: List[int]
    """Hidden layer width. The number of neurons in each hidden layer."""
    n_hidden_layers: Optional[int] = None
    """Depth. The number of hidden layers in the model. If less than the length of
    layers_hidden, the model is truncated. If greater, all layers are used.
    """

    grid_size: int = 5
    """The grid size controls the resolution of the B-spline."""
    spline_order: int = 3
    """The order of the spline controls the smoothness of the B-spline."""
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    base_activation: ActivationEnum = ActivationEnum.SiLU
    grid_eps: float = 0.02
    grid_range: Tuple[float, float] = (-1, 1)

    enable_base_weight: bool = True
    enable_standalone_scale_spline: bool = True
    cl_update_grid: bool = False

    activation_penalty_weight: float = 0
    """EfficientKAN's (Blealtan and Dash 2024) alternative to entropy based 
    regularisation (Liu et al. 2024)."""

    entropy_penalty_weight: float = 0
    """EfficientKAN's dumb simulation of the entropy based regularisation."""

    first_layer_is_linear: bool = False
    """If True, the initial layer is replaced with a linear layer. This is
    useful to reduce the dimensionality before using expensive splines."""
    last_layer_is_linear: bool = False
    """If True, the final layer is replaced with a linear layer."""

    norm: Literal["none", "batch", "layer"] = "none"
    """The type of normalization to apply before each hidden layer."""


class PackNet(ConfigBase):
    type_: Literal["PackNet"] = "PackNet"
    layers_hidden: List[int] = [400]
    n_hidden_layers: Optional[int] = None
    """The number of hidden layers in the model. If less than the length of
    layers_hidden, the model is truncated. If greater, all layers are used.
    """

    prune_ratio: float = 0.5
    """The proportion of remaining weights and biases that should be pruned."""
    prune_after_p_percent: float = Field(0.5, ge=0, le=1)
    """Prune after this percentage of training."""


class WiseKAN(EfficientKAN):
    type_: Literal["WiseKAN"] = "WiseKAN"
    start_edge_prune_percent: float = Field(0.1, ge=0, le=1)
    """The percentage after which the model starts pruning the spline edges."""
    start_coef_prune_percent: float = Field(0.5, ge=0, le=1)
    """The percentage after which the model starts pruning the coefficients."""
    early_stopping_threshold: float = Field(0.1, ge=0, le=1)
    """The percentage difference in validation metric (usually R2) that triggers
    early stopping."""
    target_sparsity: float = Field(0.95, ge=0, le=1)
    """The target sparsity that the automated gradual pruning algorithm aims to
    achieve."""
    min_sparsity: float = Field(0.1, ge=0, le=1)
    """If the model is below this sparsity pruning cannot be stopped by early stopping."""


class WiseMLP(ConfigBase):
    type_: Literal["WiseMLP"] = "WiseMLP"
    layers_hidden: List[int] = [400]
    n_hidden_layers: Optional[int] = None
    """The number of hidden layers in the model. If less than the length of
    layers_hidden, the model is truncated. If greater, an error is raised.
    """
    start_coef_prune_percent: float = Field(0.5, ge=0, le=1)
    """The percentage after which the model starts pruning weights and biases."""
    min_sparsity: float = Field(0.1, ge=0, le=1)
    early_stopping_threshold: float = Field(0.1, ge=0, le=1)
    target_sparsity: float = Field(1, ge=0, le=1)
    """If the model is below this sparsity pruning cannot be stopped by early stopping."""


class MLP(ConfigBase):
    type_: Literal["MLP"] = "MLP"
    layers_hidden: List[int] = [256, 256]
    n_hidden_layers: Optional[int] = None
    """The number of hidden layers in the model. If less than the length of
    layers_hidden, the model is truncated. If greater, an error is raised.
    """

    activation: ActivationEnum = ActivationEnum.ReLU
    output_activation: ActivationEnum = ActivationEnum.Identity


ModelTypes = EfficientKAN | MLP | WiseKAN | PackNet | WiseMLP

# Scenario ---------------------------------------------------------------------


class GaussianPeaks(ConfigBase):
    """A toy continual learning problem. The dataset samples from a sequence of
    Gaussian peaks. See Figure 3.4 in [1] for an illustration.

    References:
    -----------
    1. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M.,
    Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks
    (arXiv:2404.19756). arXiv. http://arxiv.org/abs/2404.19756
    """

    type_: Literal["GaussianPeaks"] = "GaussianPeaks"
    peaks: int = 5
    """The number of peaks in the dataset. Each peak is used as a continual
    learning task.
    """
    width: float = 1 / 4
    """The spread/std-deviation of the Gaussian peaks. Lower values make the
    peaks more narrow. The spread is divided by the width of the peaks so that
    the peaks are correct regardless of the number of peaks. 
    """
    num_test_task_samples: int = 10_000
    """The number of samples in each test task"""
    num_train_task_samples: int = 3_000
    """The number of samples in each training task"""


class IncrementalFeynman(ConfigBase):
    type_: Literal["IncrementalFeynman"] = "IncrementalFeynman"
    hdf5_filename: str
    """The filename of the HDF5 file containing the dataset."""
    shuffle: bool = False
    joint: bool = False
    """If true, the tasks are learned jointly, making the dataset non-incremental."""
    standard_score_normalize: bool = True
    """If true, normalize each dataset to have zero mean and unit variance."""
    min_max_normalize: bool = False
    """If true, normalize each dataset to have values between 0 and 1."""
    valid_proportion: float = 0.1
    """The proportion of the training data to use as validation data."""
    orthogonalize_targets: Literal[True] = True


class TaskIncrementalRegression(ConfigBase):
    type_: Literal["TaskIncrementalRegression"] = "TaskIncrementalRegression"
    test_proportion: float = 1 / 6
    """The proportion of the training data to use as test data."""
    valid_proportion: float = 1 / 6
    """The proportion of the training data to use as validation data."""
    num_tasks: int = 5
    """The number of tasks to split the dataset into."""
    hdf5_filename: str
    """The filename of the HDF5 file containing the dataset."""
    orthogonalize_inputs: bool = False
    """If true, the inputs are orthoganalized before training."""
    orthogonalize_targets: bool = False
    """If true, the outputs are orthoganalized before training."""


ScenarioTypes = GaussianPeaks | IncrementalFeynman | TaskIncrementalRegression

# Strategies -------------------------------------------------------------------


class EWC(ConfigBase):
    type_: Literal["EWC"] = "EWC"
    ewc_lambda: float = 1.0
    """The strength of the EWC regularisation."""


class SI(ConfigBase):
    type_: Literal["SI"] = "SI"
    si_lambda: float = 1.0
    """The strength of the SI regularisation."""
    epsilon: float = 0.001


StrategyTypes = EWC | SI

# Plugins ----------------------------------------------------------------------


class Training(ConfigBase):
    lr: float = 1e-3
    """Learning Rate. Defines the step size for the optimizer."""
    train_mb_size: int = 32
    eval_mb_size: Optional[int] = None
    device: str = "cuda"
    epochs: int = 10
    num_workers: int = 2
    """The number of workers used to load the dataset and apply transforms"""
    optimizer: Literal["Adam", "SGD", "LBFGS"] = "Adam"
    initial_task_epochs: Optional[int] = None
    """(Optional) Override the number of epochs for the first task. This is 
    useful to train the model on the first task for longer than subsequent tasks.
    """

    def model_post_init(self, __context: Any) -> None:
        if self.eval_mb_size is None:
            self.eval_mb_size = self.train_mb_size
        if self.initial_task_epochs is None:
            self.initial_task_epochs = self.epochs


class Config(ConfigBase):
    model: ModelTypes = Field(MLP(), discriminator="type_")
    strategy: Optional[StrategyTypes] = Field(None, discriminator="type_")
    scenario: ScenarioTypes = Field(None, discriminator="type_")
    training: Training = Training()
    tags: List[str] = []
    run_id: Optional[int] = None
    mb_log_freq: Optional[int] = 100
    """Limit the frequency with which metrics reported per mini-batch are 
    actually logged. This is useful to reduce the amount of logging."""
    target_metric: Literal["R2", "RMSE"] = "R2"
    watch_gradients: bool = False


__all__ = [
    "Config",
    "ModelTypes",
    "ScenarioTypes",
    "Training",
    "EfficientKAN",
    "MLP",
    "GaussianPeaks",
]
