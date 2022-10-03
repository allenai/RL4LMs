import re
from typing import Any, Dict

from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction


class RewardIncreasingNumbers(RewardFunction):
    def __init__(self,
                 min_tokens: int) -> None:
        super().__init__()
        self.min_tokens = min_tokens

    @staticmethod
    def is_number(text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def reward_increasing_numbers_in_text(gen_text: str,
                                          min_tokens: int):
        gen_tokens = gen_text.split()
        number_tokens = [float(token)
                         for token in gen_tokens if RewardIncreasingNumbers.is_number(token)]
        if len(number_tokens) > 0:
            # then we check how many numbers are in the sorted order
            sorted_count = 1
            previous_token = number_tokens[0]
            for token in number_tokens[1:]:
                if token > previous_token:
                    sorted_count += 1
                    previous_token = token
                else:
                    break
            return ((sorted_count)/max(len(gen_tokens), (min_tokens/2)))
        return 0

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:
            gen_text = current_observation.context_text
            reward = RewardIncreasingNumbers.reward_increasing_numbers_in_text(
                gen_text, self.min_tokens)
            return reward
        return 0


class RewardSentencesWithDates:

    def date_in_text(text: str):
        match = re.search(r'\d{4}-\d{2}-\d{2}',
                          text)
        if match is not None:
            return 1
        else:
            return 0

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool,
                 meta_info: Dict[str, Any] = None) -> float:
        if done:
            return RewardSentencesWithDates.date_in_text(current_observation.context_text)
        return 0
