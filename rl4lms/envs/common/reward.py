
from rl4lms.envs.common.observation import BaseObservation
from abc import ABC, abstractclassmethod
from typing import List


class RewardFunction(ABC):
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
