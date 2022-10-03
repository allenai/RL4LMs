
from typing import Any, Dict, List

import numpy as np
from rl4lms.envs.text_generation.metric import BaseMetric
from rl4lms.envs.text_generation.test_reward import (RewardIncreasingNumbers,
                                                     RewardSentencesWithDates)
from transformers import PreTrainedModel


class IncreasingNumbersinText(BaseMetric):
    def __init__(self, min_tokens: int) -> None:
        super().__init__()
        self._min_tokens = min_tokens

    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        for gen_text in generated_texts:
            reward = RewardIncreasingNumbers.reward_increasing_numbers_in_text(
                gen_text, self._min_tokens)
            all_rewards.append(reward)

        metric_dict = {
            "synthetic/increasing_numbers_in_text": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict


class DateInText(BaseMetric):
    def compute(self, prompt_texts: List[str],
                generated_texts: List[str],
                reference_texts: List[List[str]],
                meta_infos: List[Dict[str, Any]] = None,
                model: PreTrainedModel = None,
                split_name: str = None) -> Dict[str, float]:

        all_rewards = []
        for gen_text in generated_texts:
            reward = RewardSentencesWithDates.date_in_text(
                gen_text)
            all_rewards.append(reward)
        metric_dict = {
            "synthetic/dates_in_text": (all_rewards, np.mean(all_rewards))
        }
        return metric_dict
