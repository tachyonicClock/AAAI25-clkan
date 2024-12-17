import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def agp_schedule(
    step: int,
    initial_sparsity: float,
    target_sparsity: float,
    total_steps: int,
    start_step: int,
) -> float:
    """Automated gradual pruning schedule (AGP)

    :param initial_sparsity: The initial sparsity level
    :param target_sparsity: The target sparsity level
    :param total_prune_steps: The total number of pruning steps
    :param start_step: The step to start pruning
    :param step: The current step
    :return: The sparsity level for the current step
    """
    if step < start_step:
        return initial_sparsity
    elif step >= total_steps:
        return target_sparsity

    return (
        target_sparsity
        + (initial_sparsity - target_sparsity)
        * (1 - (step - start_step) / (total_steps - start_step)) ** 3
    )


def relative_change(reference: float, value: float) -> float:
    """Calculate the relative change between two values

    :param a: The first value
    :param b: The second value
    :return: The relative change
    """
    return (value - reference) / abs(reference)


def early_stop(
    valid_score: float,
    sparsity: float,
    min_sparsity: float,
    best_score: float,
    threshold: float,
) -> bool:
    """Determine if early stopping should be triggered.

    :param valid_score: The validation score (higher is better).
    :param sparsity: The current sparsity level.
    :param min_sparsity: The minimum sparsity level to trigger early stopping.
    :param best_score: The best validation score so far to compare against.
    :param threshold: The relative threshold to trigger early stopping.
    :return: True if early stopping should be triggered, False otherwise.
    """
    assert 0 <= min_sparsity <= 1, "min_sparsity must be in [0, 1]"
    assert 0 <= sparsity <= 1, "sparsity must be in [0, 1]"
    assert 0 <= threshold <= 1, "threshold must be in [0, 1]"
    relative_diff = relative_change(best_score, valid_score)

    logger.info(
        f"Stop Pruning? valid_score: {valid_score:.2f}({relative_diff*100:.1f}), sparsity: {sparsity:.2f}, best_score: {best_score:.2f}"
    )
    if best_score == 0 or sparsity <= min_sparsity:
        return False
    elif relative_diff > -threshold:
        return False
    return True


@dataclass
class AGPSchedule:
    initial_sparsity: float
    target_sparsity: float
    total_steps: Optional[int] = None
    start_step: Optional[int] = None

    def has_started(self, step: int) -> bool:
        assert self.start_step is not None
        return step >= self.start_step

    def __call__(self, step: int) -> float:
        assert self.total_steps is not None
        assert self.start_step is not None
        return agp_schedule(
            step,
            self.initial_sparsity,
            self.target_sparsity,
            self.total_steps,
            self.start_step,
        )


@dataclass
class EarlyStopping:
    min_sparsity: float
    threshold: float
    best_score: Optional[float] = None

    def reset(self):
        self.best_score = None

    def __call__(self, valid_score: float, sparsity: float) -> bool:
        if self.best_score is None:
            self.best_score = valid_score
        self.best_score = max(self.best_score, valid_score)
        return early_stop(
            valid_score,
            sparsity,
            self.min_sparsity,
            self.best_score,
            self.threshold,
        )
