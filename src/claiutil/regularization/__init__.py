"""Implements Regularization Continual Learning Strategies."""

from . import ewc, si
from ._common import (
    MutTensorSequence,
    copy_parameters,
    parameters_like_module,
    quadratic_penalty,
    zero_parameters,
)

__all__ = [
    "copy_parameters",
    "parameters_like_module",
    "quadratic_penalty",
    "MutTensorSequence",
    "zero_parameters",
    "si",
    "ewc",
]
