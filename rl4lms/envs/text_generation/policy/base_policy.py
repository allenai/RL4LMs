from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule, TensorDict
from torch.distributions import Categorical
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import unwrap_model


class PolicyType(Enum):
    CAUSAL = 0
    SEQ2SEQ = 1


@dataclass
class EvaluateActionsOutput:
    """
    Dataclass for the output of the method policy.evaluate_actions().
    This is invoked during training phase for each mini-batch in the rollout buffer
    """

    # values of the given state
    values: torch.tensor
    # log prob of chosen actions
    log_prob: torch.tensor
    # entropy of action dist
    entropy: torch.tensor


@dataclass
class PolicyOutput:
    """
    Dataclass for the output of the method policy.foward_policy()
    """

    # chosen actions by policy
    actions: torch.tensor
    # raw log probs corresponding to chosen actions
    raw_log_probs: torch.tensor
    # processed log probs (eg: after action masking) for chosen actions
    log_probs: torch.tensor
    # entropy of action dist
    entropy: torch.tensor
    # cached policy activations for sequential forward passes
    past_model_kwargs: torch.tensor


@dataclass
class RefPolicyOutput:
    """
    Dataclass for the output of the method policy.get_ref_log_probs()
    """

    # ref log_probs for corresponding observation and chosen action
    log_probs: torch.tensor
    # cached policy activations for sequential forward passes
    past_model_kwargs: torch.tensor


@dataclass
class ValueOutput:
    """
    Dataclass for the output of the method policy.forward_value()
    """

    # values corresponding to given state
    values: torch.tensor
    # cached value activations for sequential forward passes
    past_model_kwargs: Dict[str, torch.tensor]


@dataclass
class GenerationInputs:
    # prompt inputs
    inputs: torch.tensor
    # prompt attention masks
    attention_masks: torch.tensor


@dataclass
class GenerationOutputs:
    # log probs at each time step
    step_wise_logprobs: List[List[torch.tensor]]
    # actions at each time step
    step_wise_actions: List[torch.tensor]
    # generated tokens
    gen_tokens: List[List[int]]
    # generated texts
    gen_texts: List[str]
    # action masks
    action_masks: List[torch.tensor] = None


class LMActorCriticPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        lr_schedule: Schedule,
        model_name: str,
        optimizer_kwargs: Dict[str, Any] = {},
        weight_decay: float = 1e-6,
        use_sde: bool = None,
        apply_model_parallel: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
    ):
        """

        Args:
            observation_space (DictSpace): Observation space
            action_space (Discrete): Action space
            lr_schedule (Schedule): Learning rate schedule
            model_name (str): name of the causal or seq2seq model from transformers library
            optimizer_kwargs (Dict[str, Any], optional): optimizer kwargs. Defaults to {}.
            weight_decay (float, optional): weight decay. Defaults to 1e-6.
            use_sde (bool, optional): Use state-dependent exploration. Defaults to None. (Unused parameter from stable-baselines3)
            apply_model_parallel (bool, optional): whether to apply model parallel. Defaults to True.
            optimizer_class (torch.optim.Optimizer, optional): Optimizer class. Defaults to torch.optim.AdamW.
            generation_kwargs (Dict[str, Any], optional): generation parameters for rollout. Defaults to {}.
            prompt_truncation_side (str, optional): truncation side for prompt text. Defaults to "left".
        """
        super().__init__(observation_space, action_space)
        self._action_space = action_space
        self._apply_model_parallel = apply_model_parallel
        self._build_model_heads(model_name)
        self._setup_optimizer(optimizer_kwargs, weight_decay, optimizer_class)
        self._action_dist = CategoricalDistribution(self._action_space.n)
        self._generation_kwargs = generation_kwargs
        self._prompt_truncation_side = prompt_truncation_side

    def _setup_optimizer(
        self,
        optimizer_kwargs: Dict[str, Any],
        weight_decay: float,
        optimizer_class: torch.optim,
    ):
        params = list(self.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_kwargs
        )

    def forward(self, *args, **kwargs):
        # dummy just to comply with base policy
        pass

    @staticmethod
    def _predict(
        self, observation: Dict[str, torch.tensor], deterministic: bool = False
    ) -> torch.Tensor:
        # dummy just to comply with base policy
        pass

    def is_encoder_decoder(self, model: PreTrainedModel):
        return unwrap_model(model).config.is_encoder_decoder

    def generate(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str] = None,
        max_prompt_length: int = None,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        gen_kwargs: Dict[str, Any] = None,
    ) -> GenerationOutputs:

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self._generation_kwargs

        # switch to eval
        self._policy_model.eval()

        if (
            input_ids is None
            and attention_mask is None
            and texts is not None
            and max_prompt_length is not None
        ):
            # override truncation side for prompt
            prev_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = self._prompt_truncation_side
            encodings = tokenizer(
                texts,
                padding="max_length",
                max_length=max_prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            tokenizer.truncation_side = prev_truncation_side

        # if min_length argument is set and if policy is not a seq2seq LM (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if "min_length" in gen_kwargs.keys() and not self.is_encoder_decoder(
            self._policy_model
        ):
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_["min_length"] = (
                input_ids.shape[1] + gen_kwargs["min_length"]
            )
        else:
            generation_kwargs_ = gen_kwargs

        # generate
        gen_output = unwrap_model(self._policy_model).generate(
            inputs=input_ids.to(self.get_policy_first_device()),
            attention_mask=attention_mask.to(self.get_policy_first_device()),
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs_,
        )

        # number of tokens generated
        seq_length = len(gen_output["scores"])

        # get only the generated text (excluding prompt)
        gen_tokens = gen_output["sequences"][:, -seq_length:]

        # to texts
        gen_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in gen_tokens.tolist()
        ]

        # extract scores (logits)
        step_wise_logprobs = []
        step_wise_actions = []
        for step, logits in enumerate(gen_output["scores"]):
            raw_logits, _ = logits
            actions_at_step = gen_tokens[:, step]
            distribution = Categorical(logits=raw_logits)
            log_probs = distribution.log_prob(actions_at_step)
            step_wise_logprobs.append(log_probs)
            step_wise_actions.append(actions_at_step)

        gen_output = GenerationOutputs(
            step_wise_logprobs, step_wise_actions, gen_tokens, gen_texts
        )
        return gen_output

    def get_language_model(self):
        return unwrap_model(self._policy_model)

    # Following methods need to be implemented by sub-classing
    @abstractmethod
    def _build_model_heads(self, model_name: str):
        """
        Builds policy and value models
        and sets self._policy_model and self._value_model
        """
        raise NotImplementedError

    @abstractmethod
    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:
        """
        Performs a forward pass on the policy and gets log_probs, entropy etc
        corresponding to specified observation, actions

        This is invoked during rollout generation

        Args:
            obs (TensorDict): observation
            actions (torch.tensor): actions
            past_model_kwargs (Optional[Dict[str, torch.tensor]], optional): Any cached past model activations which can be used for sequential foward passes.
            Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_value(
        self,
        obs: TensorDict,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:
        """
        Performs a forward pass on the value network and gets values corresponding to observations

        This is invoked during rollout generation

        Args:
            obs (TensorDict): observation
            past_model_kwargs (Optional[Dict[str, torch.tensor]], optional): Any cached past model activations which can be used for sequential foward passes.
            Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> EvaluateActionsOutput:
        """
        Evaluates specified <observation, action>
        and returns log_probs, values, entropy

        This is invoked for each mini-batch in rollout buffer during training iteration
        """
        raise NotImplementedError

    @abstractmethod
    def get_log_probs_ref_model(
        self,
        obs: TensorDict,
        action: torch.tensor,
        past_model_kwargs: Dict[str, Any] = None,
    ) -> RefPolicyOutput:
        """
        Performs a forward pass on the reference policy and gets log_probs
        corresponding to specified observation, actions

        This is invoked during rollout generation to compute KL rewards

        Args:
            obs (TensorDict): observation
            past_model_kwargs (Optional[Dict[str, torch.tensor]], optional): Any cached past model activations which can be used for sequential foward passes.
            Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_first_device(self) -> torch.device:
        """
        Returns the first device of the policy. Used in the case of model parallel
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_type(self) -> PolicyType:
        """
        Returns the type of policy (causal or seq2seq)
        """
        raise NotImplementedError

    @abstractmethod
    def get_inputs_for_generation(self, obs: TensorDict) -> GenerationInputs:
        """
        Extracts the prompt inputs and attention masks which is used as seed for generation
        """
        raise NotImplementedError
