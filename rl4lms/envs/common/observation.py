from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch


@dataclass
class BaseObservation:
    """
    Placeholder for observation data class
    """
    pass


class BaseObservationFeaturizer(ABC):

    @abstractmethod
    def featurize(self, observation: BaseObservation) -> torch.Tensor:
        raise NotImplementedError

    def get_observation_dim(self) -> int:
        """
        Returns the observation dim
        """
        return self.get_input_dim() + self.get_context_dim()