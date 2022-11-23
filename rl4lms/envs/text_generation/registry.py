from typing import Any, Dict, Type, Union

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from rl4lms.algorithms.a2c.a2c import A2C
from rl4lms.algorithms.nlpo import NLPO
from rl4lms.algorithms.ppo.ppo import PPO
from rl4lms.algorithms.trpo import TRPO
from rl4lms.data_pools.custom_text_generation_pools import (
    IMDB,
    WMT,
    CNNDailyMail,
    CommonGen,
    CRD3DialogueGeneration,
    IMDBForSeq2Seq,
    IWSLT2017EnDe,
    NarrativeQA,
    ToTTo,
    WMT14PreprocessedEnDe,
    WMT16NewsOnlyDatasetEnDe,
    DailyDialog,
)
from rl4lms.data_pools.text_generation_pool import TextGenPool
from rl4lms.envs.text_generation.alg_wrappers import wrap_onpolicy_alg
from rl4lms.envs.text_generation.metric import (
    BaseMetric,
    BERTScoreMetric,
    BLEUMetric,
    BLEURTMetric,
    BLEUToTTo,
    CIDERMetric,
    DiversityMetrics,
    LearnedRewardMetric,
    MeteorMetric,
    ParentToTTo,
    Perplexity,
    RougeLMax,
    RougeMetric,
    SacreBLEUMetric,
    SpiceMetric,
    SummaCConvMetric,
    SummaCZSMetric,
    TERMetric,
    chrFmetric,
    IntentAccuracyDailyDialog,
)
from rl4lms.envs.text_generation.policy.base_policy import LMActorCriticPolicy
from rl4lms.envs.text_generation.policy.causal_policy import (
    CausalLMActorCriticPolicy,
    MaskedCausalLMActorCriticPolicy,
)
from rl4lms.envs.text_generation.policy.seq2seq_policy import (
    Seq2SeqLMActorCriticPolicy,
    MaskedSeq2SeqLMActorCriticPolicy,
)
from rl4lms.envs.text_generation.post_processors import three_sentence_summary
from rl4lms.envs.text_generation.reward import (
    TER,
    BatchedCommonGenPenaltyShapingFunction,
    BERTScoreRewardFunction,
    BLEURewardFunction,
    BLEURTRewardFunction,
    CommonGenPenaltyShapingFunction,
    LearnedRewardFunction,
    MeteorRewardFunction,
    PARENTRewardFunction,
    RewardFunction,
    RougeCombined,
    RougeLMaxRewardFunction,
    RougeRewardFunction,
    SacreBleu,
    SpiderRewardFunction,
    chrF,
    IntentAccuracy,
)
from rl4lms.envs.text_generation.preference_reward import CommonGenPrefRM
from rl4lms.envs.text_generation.test_datapool import TestTextGenPool
from rl4lms.envs.text_generation.test_metric import DateInText, IncreasingNumbersinText
from rl4lms.envs.text_generation.test_reward import (
    RewardIncreasingNumbers,
    RewardSentencesWithDates,
)


class DataPoolRegistry:
    _registry = {
        "dummy_pool": TestTextGenPool,
        "imdb": IMDB,
        "commongen": CommonGen,
        "totto": ToTTo,
        "cnn_daily_mail": CNNDailyMail,
        "imdb_seq2seq": IMDBForSeq2Seq,
        "narrative_qa": NarrativeQA,
        "wmt16": WMT,
        "wmt14_processed_en_de": WMT14PreprocessedEnDe,
        "wmt16newsonly": WMT16NewsOnlyDatasetEnDe,
        "iwslt2017en_de": IWSLT2017EnDe,
        "crd3": CRD3DialogueGeneration,
        "daily_dialog": DailyDialog,
    }

    @classmethod
    def get(cls, datapool_id: str, kwargs: Dict[str, Any]) -> TextGenPool:
        datapool_cls = cls._registry[datapool_id]
        datapool = datapool_cls.prepare(**kwargs)
        return datapool

    @classmethod
    def add(cls, id: str, datapool_cls: Type[TextGenPool]):
        DataPoolRegistry._registry[id] = datapool_cls


class RewardFunctionRegistry:
    _registry = {
        "increasing_numbers": RewardIncreasingNumbers,
        "sentences_with_dates": RewardSentencesWithDates,
        "learned_reward": LearnedRewardFunction,
        "meteor": MeteorRewardFunction,
        "rouge": RougeRewardFunction,
        "bert_score": BERTScoreRewardFunction,
        "bleu": BLEURewardFunction,
        "bleurt": BLEURTRewardFunction,
        "rouge_combined": RougeCombined,
        "spider": SpiderRewardFunction,
        "common_gen_repeat_penalty": CommonGenPenaltyShapingFunction,
        "common_gen_repeat_penalty_batched": BatchedCommonGenPenaltyShapingFunction,
        "parent": PARENTRewardFunction,
        "sacre_bleu": SacreBleu,
        "rouge_l_max": RougeLMaxRewardFunction,
        "ter": TER,
        "chrf": chrF,
        "intent_accuracy": IntentAccuracy,
        "common_gen_preference_model": CommonGenPrefRM,
    }

    @classmethod
    def get(cls, reward_fn_id: str, kwargs: Dict[str, Any]) -> RewardFunction:
        reward_cls = cls._registry[reward_fn_id]
        reward_fn = reward_cls(**kwargs)
        return reward_fn

    @classmethod
    def add(cls, id: str, reward_fn_cls: Type[RewardFunction]):
        RewardFunctionRegistry._registry[id] = reward_fn_cls


class MetricRegistry:
    _registry = {
        "increasing_numbers": IncreasingNumbersinText,
        "dates": DateInText,
        "learned_reward": LearnedRewardMetric,
        "meteor": MeteorMetric,
        "rouge": RougeMetric,
        "bert_score": BERTScoreMetric,
        "bleu": BLEUMetric,
        "bleurt": BLEURTMetric,
        "diversity": DiversityMetrics,
        "summaCZS": SummaCZSMetric,
        "summaCConv": SummaCConvMetric,
        "causal_perplexity": Perplexity,
        "cider": CIDERMetric,
        "spice": SpiceMetric,
        "parent_totto": ParentToTTo,
        "bleu_totto": BLEUToTTo,
        "rouge_l_max": RougeLMax,
        "sacre_bleu": SacreBLEUMetric,
        "ter": TERMetric,
        "chrf": chrFmetric,
        "intent_accuracy": IntentAccuracyDailyDialog,
    }

    @classmethod
    def get(cls, metric_id: str, kwargs: Dict[str, Any]) -> BaseMetric:
        metric_cls = cls._registry[metric_id]
        metric = metric_cls(**kwargs)
        return metric

    @classmethod
    def add(cls, id: str, metric_cls: Type[BaseMetric]):
        MetricRegistry._registry[id] = metric_cls


class PolicyRegistry:
    _registry = {
        "causal_lm_actor_critic_policy": CausalLMActorCriticPolicy,
        "seq2seq_lm_actor_critic_policy": Seq2SeqLMActorCriticPolicy,
        "maskable_causal_lm_actor_critic_policy": MaskedCausalLMActorCriticPolicy,
        "maskable_seq2seq_lm_actor_critic_policy": MaskedSeq2SeqLMActorCriticPolicy,
    }

    @classmethod
    def get(cls, policy_id: str) -> Type[LMActorCriticPolicy]:
        policy_cls = cls._registry[policy_id]
        return policy_cls

    @classmethod
    def add(cls, id: str, policy_cls: Type[LMActorCriticPolicy]):
        PolicyRegistry._registry[id] = policy_cls


class AlgorithmRegistry:
    _registry = {
        "nlpo": NLPO,
        "trpo": TRPO,
        "ppo": PPO,
        "a2c": A2C,
    }

    @classmethod
    def get(
        cls, alg_id: str
    ) -> Union[Type[OnPolicyAlgorithm], Type[OffPolicyAlgorithm]]:
        try:
            alg_cls = cls._registry[alg_id]
        except KeyError:
            raise NotImplementedError
        return alg_cls

    @classmethod
    def add(
        cls, id: str, alg_cls: Union[Type[OnPolicyAlgorithm], Type[OffPolicyAlgorithm]]
    ):
        AlgorithmRegistry._registry[id] = alg_cls


class WrapperRegistry:
    _registry = {
        "nlpo": wrap_onpolicy_alg,
        "trpo": wrap_onpolicy_alg,
        "ppo": wrap_onpolicy_alg,
        "a2c": wrap_onpolicy_alg,
    }

    @classmethod
    def get(cls, alg_id: str):
        try:
            wrapper_def = cls._registry[alg_id]
        except KeyError:
            raise NotImplementedError
        return wrapper_def

    @classmethod
    def add(cls, id: str, wrapper_def):
        WrapperRegistry._registry[id] = wrapper_def


class PostProcessorRegistry:
    _registry = {
        "three_sentence_summary": three_sentence_summary,
    }

    @classmethod
    def get(cls, post_processor_id: str):
        try:
            wrapper_def = cls._registry[post_processor_id]
        except KeyError:
            raise NotImplementedError
        return wrapper_def

    @classmethod
    def add(cls, id: str, post_processor_fn):
        PostProcessorRegistry._registry[id] = post_processor_fn
