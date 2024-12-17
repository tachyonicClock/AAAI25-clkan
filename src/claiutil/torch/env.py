"""Utilities for setting up the environment for PyTorch."""

from os import environ
from pathlib import Path
from random import seed as python_seed

import torch
from numpy.random import seed as numpy_seed
from torch import manual_seed as torch_seed
from torch.cuda import manual_seed as torch_cuda_seed


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility."""
    torch_seed(seed)
    numpy_seed(seed)
    python_seed(seed)
    if torch.cuda.is_available():
        torch_cuda_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


def torch_data_dir() -> str:
    """Get the root directory for PyTorch."""
    torch_data_dir_ = environ.get("TORCH_DATA_DIR")
    if not torch_data_dir_:
        raise RuntimeError("The ``TORCH_DATA_DIR`` environment variable should be set.")
    torch_data_dir_path = Path(torch_data_dir_).expanduser().resolve()
    torch_data_dir_path.mkdir(exist_ok=True)
    return str(torch_data_dir_path)


__all__ = ["set_seed", "torch_data_dir"]
