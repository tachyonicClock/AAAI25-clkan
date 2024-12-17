from torch import nn


class HasTaskId:
    _task_id: int = 0

    def set_task_id(self, task_id: int, recursive: bool = False) -> None:
        self._task_id = task_id
        if recursive and isinstance(self, nn.Module):
            for module in self.modules():
                if isinstance(module, HasTaskId):
                    module.set_task_id(task_id)

    def get_task_id(self) -> int:
        return self._task_id
