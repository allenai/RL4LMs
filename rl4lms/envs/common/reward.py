
from rl4lms.envs.common.observation import BaseObservation
from abc import ABC, abstractclassmethod
from accelerate import Accelerator
from typing import List


class RewardFunction(ABC):
    def __init__(self, accelerator: Accelerator) -> None:
        super().__init__()
        self._accelerator = accelerator

    @abstractclassmethod
    def __call__(self, observation: BaseObservation, action: str, targets: List[str]) -> float:
        """[summary]

        Args:
            observation (Observation): current observation at t
            action (str): current action at t
            targets (List[str]): targets of the current sample

        Returns:
            - a scalar reward
        """
        raise NotImplementedError
