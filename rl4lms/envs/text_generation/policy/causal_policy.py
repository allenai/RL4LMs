from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule, TensorDict
from torch import nn
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import unwrap_model

from rl4lms.algorithms.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)
from rl4lms.algorithms.common.maskable.logits_processor import (
    MaskLogitsProcessorCasualLM,
)
from rl4lms.envs.text_generation.hf_generation_utils import override_generation_routines
from rl4lms.envs.text_generation.policy.base_policy import (
    EvaluateActionsOutput,
    GenerationInputs,
    GenerationOutputs,
    LMActorCriticPolicy,
    PolicyOutput,
    PolicyType,
    RefPolicyOutput,
    ValueOutput,
)
from rl4lms.envs.text_generation.warm_start import (
    ActorCriticWarmStartMixin,
    MaskableActorCriticWarmStartMixin,
)


class CausalLMActorCriticPolicy(LMActorCriticPolicy, ActorCriticWarmStartMixin):
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
        state_dict: Dict[str, Any] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            model_name,
            optimizer_kwargs,
            weight_decay,
            use_sde,
            apply_model_parallel,
            optimizer_class,
            generation_kwargs,
            prompt_truncation_side,
        )
        self.load_from_dict(state_dict)

    def _build_model_heads(self, model_name: str):
        self._policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        self._policy_model.__class__ = override_generation_routines(
            type(self._policy_model)
        )

        self._value_model = AutoModelForCausalLM.from_pretrained(model_name)
        self._ref_model = deepcopy(self._policy_model).eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False
        )

        # apply model parallel
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
                self._value_model.parallelize()
                self._value_head = self._value_head.to(self.device)
            else:  # else defaults to data parallel
                self._policy_model = torch.nn.DataParallel(self._policy_model)
                self._ref_model = torch.nn.DataParallel(self._ref_model)
                self._value_model = torch.nn.DataParallel(self._value_model)
                self._value_head = torch.nn.DataParallel(
                    self._value_head.to(self.device)
                )

    def _prepare_inputs_for_model(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        model_inputs = unwrap_model(model).prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )

        if self._apply_model_parallel and unwrap_model(model).is_parallelizable:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {
                key: value.to(model.transformer.first_device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in model_inputs.items()
            }
        return model_inputs

    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            # take attention mask only for the first step
            # for subsequent steps, update_model_kwargs will handle it
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._policy_model, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self._policy_model(output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        entropy = dist.entropy()

        # sample act
        log_prob = dist.log_prob(actions)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._policy_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._policy_model
            ).config.is_encoder_decoder,
        )

        policy_outputs = PolicyOutput(
            actions=actions,
            raw_log_probs=log_prob,
            log_probs=log_prob,
            entropy=entropy,
            past_model_kwargs=past_model_kwargs,
        )

        return policy_outputs

    def forward_value(
        self,
        obs: TensorDict,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:

        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._value_model, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self._value_model(output_hidden_states=True, **model_inputs)

        # pool the hidden states ?
        last_tokens_hidden = output.hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._value_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._value_model
            ).config.is_encoder_decoder,
        )

        value_outputs = ValueOutput(values=values, past_model_kwargs=past_model_kwargs)

        return value_outputs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> EvaluateActionsOutput:

        policy_outputs = self.forward_policy(obs=obs, actions=actions)
        value_outputs = self.forward_value(obs)

        eval_outputs = EvaluateActionsOutput(
            values=value_outputs.values,
            log_prob=policy_outputs.log_probs,
            entropy=policy_outputs.entropy,
        )
        return eval_outputs

    def get_log_probs_ref_model(
        self,
        obs: TensorDict,
        action: torch.tensor,
        past_model_kwargs: Dict[str, Any] = None,
    ) -> RefPolicyOutput:
        self._ref_model = self._ref_model.eval()

        input_ids = obs["input_encoded_pt"]
        attention_mask = obs["input_attention_mask_pt"]

        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._ref_model, input_ids, past_model_kwargs
        )
        output = self._ref_model(output_hidden_states=True, **model_inputs)
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        log_prob = dist.log_prob(action)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._ref_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=self.is_encoder_decoder(self._ref_model),
        )

        ref_policy_outputs = RefPolicyOutput(log_prob, past_model_kwargs)
        return ref_policy_outputs

    def get_policy_first_device(self):
        return (
            self._policy_model.transformer.first_device
            if self._apply_model_parallel
            and unwrap_model(self._policy_model).is_parallelizable
            else "cuda"
        )

    def get_inputs_for_generation(self, obs: TensorDict):
        gen_inputs = GenerationInputs(
            obs["input_encoded_pt"], obs["input_attention_mask_pt"]
        )
        return gen_inputs

    def get_policy_type(self):
        return PolicyType.CAUSAL

    def to(self, device: str):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)
        
    def get_distribution(self, obs: TensorDict, detach=False):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        past_model_kwargs = {
            "attention_mask": attention_mask,
        }

        if detach:
            with torch.no_grad():
                model_inputs = self._prepare_inputs_for_model(
                    self._policy_model, input_ids, past_model_kwargs
                )

                # forward pass to transformers
                output = self._policy_model(output_hidden_states=True, **model_inputs)
        else:
            model_inputs = self._prepare_inputs_for_model(
                    self._policy_model, input_ids, past_model_kwargs
                )

            # forward pass to transformers
            output = self._policy_model(output_hidden_states=True, **model_inputs)


        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        return dist
    
    def predict_values(self, obs: TensorDict):
        return self.forward_value(obs).values

class MaskedCausalLMActorCriticPolicy(
    CausalLMActorCriticPolicy, MaskableActorCriticWarmStartMixin
):
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
        optimizer_class: torch.optim = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        top_mask: Union[int, float] = None,
        mask_type: str = "learned_top_k",
        target_update_iterations: int = 1000,
        prompt_truncation_side: str = "left",
        state_dict: Dict[str, Any] = None,
        min_tokens_to_keep: int = 100,
    ):
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mask_type = mask_type
        self.top_mask = top_mask if top_mask != -1 else self._action_space.n
        self.target_update_iterations = target_update_iterations
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            model_name,
            optimizer_kwargs,
            weight_decay,
            use_sde,
            apply_model_parallel,
            optimizer_class,
            generation_kwargs,
            prompt_truncation_side,
            state_dict,
        )

        self._action_dist = MaskableCategoricalDistribution(self._action_space.n)
        self._ref_action_dist = CategoricalDistribution(self._action_space.n)
        self._mask_action_dist = CategoricalDistribution(self._action_space.n)
        self.all_special_ids = None

    def _build_model_heads(self, model_name: str):
        super()._build_model_heads(model_name)
        if "learned" in self.mask_type:
            self._mask_model = deepcopy(self._policy_model).eval()
        else:
            self._mask_model = self._ref_model.eval()

        if torch.cuda.is_available():
            if (
                unwrap_model(self._mask_model).is_parallelizable
                and self._apply_model_parallel
            ):
                self._mask_model.parallelize()
            else:
                self._mask_model = torch.nn.DataParallel(self._mask_model)

        self.logits_processor = MaskLogitsProcessorCasualLM(
            self._mask_model,
            self.action_space,
            self.top_mask,
            self._apply_model_parallel,
            self.get_policy_first_device,
            self.mask_type,
            self.min_tokens_to_keep,
        )

    def _get_action_masks(
        self, input_ids: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        action_masks = torch.zeros((input_ids.size(0), self.action_space.n)).to(
            self.device
        )
        model_kwargs = {
            "attention_mask": attention_mask,
        }
        model_inputs = self._prepare_inputs_for_model(
            self._mask_model, input_ids, model_kwargs
        )
        output = self._mask_model(output_hidden_states=True, **model_inputs)

        next_token_logits = output.logits[:, -1, :]
        ref_distr = self._mask_action_dist.proba_distribution(
            action_logits=next_token_logits
        )
        next_token_probs = ref_distr.distribution.probs
        _, topk_indices = torch.topk(
            next_token_probs, k=self.top_mask, dim=1, sorted=True
        )
        action_masks = action_masks.scatter(index=topk_indices.long(), dim=1, value=1)

        if self.all_special_ids is not None:
            action_masks = action_masks.scatter(
                index=self.all_special_ids, dim=1, value=1
            )

        action_masks = action_masks.bool()

        return action_masks

    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        action_masks: torch.Tensor = None,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            # take attention mask only for the first step
            # for subsequent steps, update_model_kwargs will handle it
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._policy_model, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self._policy_model(output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)

        # get raw logits
        raw_logprobs = dist.log_prob(actions)

        if action_masks is None:
            action_masks = self._get_action_masks(input_ids, attention_mask)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        entropy = dist.entropy()

        # sample act
        log_prob = dist.log_prob(actions)
        # assert torch.all(torch.isfinite(log_prob))

        # update the model kwargs for further generation
        past_model_kwargs = self._policy_model._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=self._policy_model.config.is_encoder_decoder,
        )

        policy_output = PolicyOutput(
            actions=actions,
            raw_log_probs=raw_logprobs,
            log_probs=log_prob,
            entropy=entropy,
            past_model_kwargs=past_model_kwargs,
        )

        return policy_output

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, action_masks: torch.Tensor
    ) -> EvaluateActionsOutput:

        policy_outputs = self.forward_policy(
            obs=obs, actions=actions, action_masks=action_masks
        )
        value_outputs = self.forward_value(obs)

        eval_outputs = EvaluateActionsOutput(
            values=value_outputs.values,
            log_prob=policy_outputs.log_probs,
            entropy=policy_outputs.entropy,
        )
        return eval_outputs

    def generate(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str] = None,
        max_prompt_length: int = None,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        gen_kwargs: Dict[str, Any] = None,
    ):

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self._generation_kwargs

        # switch to eval
        self._policy_model.eval()
        self.logits_processor.reset()

        if (
            input_ids is None
            and attention_mask is None
            and texts is not None
            and max_prompt_length is not None
        ):
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

        self.logits_processor.attention_mask = attention_mask.to(
            self.get_policy_first_device()
        )
        self.logits_processor.all_special_ids = self.all_special_ids = (
            torch.tensor(
                tokenizer.all_special_ids,
                dtype=input_ids.dtype,
                device=self.get_policy_first_device(),
            )
            .unsqueeze(0)
            .expand((input_ids.size(0), -1))
        )

        # if min_length argument is set and if policy is not a seq2seq LM (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if (
            "min_length" in gen_kwargs.keys()
            and not unwrap_model(self._policy_model).config.is_encoder_decoder
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
            logits_processor=[self.logits_processor],
            **generation_kwargs_
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
        action_masks = []
        for step, logits in enumerate(gen_output["scores"]):
            raw_logits, processed_logits = logits
            actions_at_step = gen_tokens[:, step]
            distribution = Categorical(logits=raw_logits)
            log_probs = distribution.log_prob(actions_at_step)
            step_wise_logprobs.append(log_probs)
            step_wise_actions.append(actions_at_step)

            # TBD: workaround due to beam search not returning processed logits yet
            if processed_logits is not None:
                # recalculating action masks
                action_mask = ~torch.isneginf(processed_logits)
                # assert torch.sum(~action_mask.long()).item() != 0
                # assert torch.all(torch.isfinite(Categorical(logits=processed_logits).log_prob(actions_at_step)))
                action_masks.append(action_mask)

        gen_output = GenerationOutputs(
            step_wise_logprobs, step_wise_actions, gen_tokens, gen_texts, action_masks
        )
        return gen_output

    def update_mask_model(self):
        self._mask_model = deepcopy(self._policy_model).eval()
