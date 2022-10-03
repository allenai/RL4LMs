import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

EXPECTED_METHOD_NAME = "action_masks"


def get_action_masks(env: GymEnv) -> np.ndarray:
    """
    Checks whether gym env exposes a method returning invalid action masks

    :param env: the Gym environment to get masks from
    :return: A numpy array of the masks
    """

    if isinstance(env, VecEnv):
        return np.stack(env.env_method(EXPECTED_METHOD_NAME))
    else:
        return getattr(env, EXPECTED_METHOD_NAME)()


def is_masking_supported(env: GymEnv) -> bool:
    """
    Checks whether gym env exposes a method returning invalid action masks

    :param env: the Gym environment to check
    :return: True if the method is found, False otherwise
    """

    if isinstance(env, VecEnv):
        try:
            # TODO: add VecEnv.has_attr()
            env.get_attr(EXPECTED_METHOD_NAME)
            return True
        except AttributeError:
            return False
    else:
        return hasattr(env, EXPECTED_METHOD_NAME)
