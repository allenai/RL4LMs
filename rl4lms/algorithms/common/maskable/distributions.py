"""
Code adapted from https://github.com/DLR-RM/stable-baselines3
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch as th
from gym import spaces
from stable_baselines3.common.distributions import Distribution
from torch import nn
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs, probs_to_logits


class MaskableCategorical(Categorical):
    """
    Modified PyTorch Categorical distribution with support for invalid action masking.

    To instantiate, must provide either probs or logits, but not both.

    :param probs: Tensor containing finite non-negative values, which will be renormalized
        to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods like lob_prob()
        and icdf() match the distribution's shape, support, etc.
    :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set to a
        large negative value, resulting in near 0 probability.
    """

    def __init__(
        self,
        probs: Optional[th.Tensor] = None,
        logits: Optional[th.Tensor] = None,
        validate_args: Optional[bool] = None,
        masks: Optional[np.ndarray] = None,
    ):
        self.masks: Optional[th.Tensor] = None
        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits
        self._original_probs = self.probs
        self.apply_masking(masks)

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(self.logits.shape)
            BIG_ZERO = th.tensor(0, dtype=self.logits.dtype, device=device)

            # Using zero's for probs instead of big negative no. for logits gives greater num stability
            probs = th.where(self.masks, self._original_probs, BIG_ZERO)

            super().__init__(probs=probs)
            self.logits = probs_to_logits(probs)
            # ^^ clamps the probs which we need for Simplex satisfaction on cross entropy loss, we want it to go to -inf only for the HF generator
            # self.logits = torch.log(probs)
        else:
            self.masks = None
        # if masks is not None:
        #     device = self.logits.device
        #     self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(self.logits.shape)
        #     HUGE_NEG = th.tensor(float('-inf'), dtype=self.logits.dtype, device=device)
        #
        #     logits = th.where(self.masks, self._original_logits, HUGE_NEG)
        # else:
        #     self.masks = None
        #     logits = self._original_logits
        #
        #     # Reinitialize with updated logits
        # super().__init__(logits=logits)
        #
        # # self.probs may already be cached, so we must force an update
        # self.probs = logits_to_probs(self.logits)

    def entropy(self) -> th.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, device=device))
        return -p_log_p.sum(-1)


class MaskableDistribution(Distribution, ABC):
    @abstractmethod
    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        """
        Eliminate ("mask out") chosen distribution outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """


class MaskableCategoricalDistribution(MaskableDistribution):
    """
    Categorical distribution for discrete actions. Supports invalid action masking.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.distribution: Optional[MaskableCategorical] = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "MaskableCategoricalDistribution":
        self.distribution = MaskableCategorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        assert self.distribution is not None, "Must set distribution parameters"
        self.distribution.apply_masking(masks)


class MaskableMultiCategoricalDistribution(MaskableDistribution):
    """
    MultiCategorical distribution for multi discrete actions. Supports invalid action masking.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.distributions: List[MaskableCategorical] = []
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "MaskableMultiCategoricalDistribution":
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, sum(self.action_dims))

        self.distributions = [
            MaskableCategorical(logits=split) for split in th.split(reshaped_logits, tuple(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        # Restructure shape to align with each categorical
        actions = actions.view(-1, len(self.action_dims))

        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([dist.sample() for dist in self.distributions], dim=1)

    def mode(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        split_masks = [None] * len(self.distributions)
        if masks is not None:
            masks = th.as_tensor(masks)

            # Restructure shape to align with logits
            masks = masks.view(-1, sum(self.action_dims))

            # Then split columnwise for each discrete action
            split_masks = th.split(masks, tuple(self.action_dims), dim=1)

        for distribution, mask in zip(self.distributions, split_masks):
            distribution.apply_masking(mask)


class MaskableBernoulliDistribution(MaskableMultiCategoricalDistribution):
    """
    Bernoulli distribution for multibinary actions. Supports invalid action masking.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dim: int):
        # Two states per binary action
        action_dims = [2] * action_dim
        super().__init__(action_dims)


def make_masked_proba_distribution(action_space: spaces.Space) -> MaskableDistribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :return: the appropriate Distribution object
    """

    if isinstance(action_space, spaces.Discrete):
        return MaskableCategoricalDistribution(action_space.n)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MaskableMultiCategoricalDistribution(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        return MaskableBernoulliDistribution(action_space.n)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Discrete, MultiDiscrete."
        )
