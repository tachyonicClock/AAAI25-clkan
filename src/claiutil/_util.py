from contextlib import contextmanager
from typing import Generator, TypeVar

from torch import nn

T = TypeVar("T", bound=nn.Module)


@contextmanager
def evaluation(module: T) -> Generator[T, None, None]:
    """Temporarily set the module to evaluation mode.

    Switching in and out of evaluation mode can be forgotten, so this context
    manager ensures that the module is in evaluation mode during the context and
    is returned to its original state afterwards.

    :param module: The module to temporarily set to evaluation mode.
    :return: The module, now in evaluation mode.
    """
    is_training = module.training
    try:
        module.eval()
        yield module
    finally:
        if is_training:
            module.train()
