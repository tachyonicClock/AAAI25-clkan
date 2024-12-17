from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


class WiseNet(ABC):
    @abstractmethod
    def regularization_loss(
        self, task_id: int, regularize_activation=1.0, regularize_entropy=1.0
    ) -> Tensor:
        pass

    @abstractmethod
    def spline_prune(
        self,
        task_id: int,
        sparsity: float,
        importance_override: Optional[Tensor] = None,
    ):
        pass

    @abstractmethod
    def coef_prune(
        self,
        task_id: int,
        sparsity: float,
        importance_override: Optional[Tensor] = None,
    ):
        pass

    @abstractmethod
    def coef_global_sparsity(self, unused_task_id: int) -> float:
        pass

    @abstractmethod
    def coef_task_sparsity(self, task_id: int) -> float:
        pass

    @abstractmethod
    def spline_global_sparsity(self, unused_task_id: int) -> float:
        pass

    @abstractmethod
    def spline_task_sparsity(self, task_id: int) -> float:
        pass
