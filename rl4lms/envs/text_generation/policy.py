from enum import Enum
from typing import Any, Dict, Optional, Tuple, List, Union
import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.distributions import Categorical
from copy import deepcopy
from rl4lms.algorithms.common.maskable.distributions import MaskableCategoricalDistribution
from rl4lms.envs.text_generation.hf_generation_utils import override_generation_routines
from stable_baselines3.common.type_aliases import TensorDict
from rl4lms.algorithms.common.maskable.logits_processor import MaskLogitsProcessorCasualLM, MaskLogitsProcessorSeq2SeqLM
from rl4lms.envs.text_generation.warm_start import ActorCriticWarmStartMixin, ActorOnlyWarmStartMixin, MaskableActorCriticWarmStartMixin
from transformers.modeling_utils import unwrap_model

class PolicyType(Enum):
    CAUSAL = 0
    SEQ2SEQ = 1


class LMActorCriticPolicy(BasePolicy, ActorCriticWarmStartMixin):
    def __init__(self, observation_space: DictSpace,
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
                 state_dict: Dict[str, Any] = None
                 ):
        super().__init__(observation_space, action_space)
        self._action_space = action_space
        self._apply_model_parallel = apply_model_parallel
        self._build_model_heads(model_name)
        self._setup_optimizer(optimizer_kwargs, weight_decay, optimizer_class)
        self.load_from_dict(state_dict)
        self._action_dist = CategoricalDistribution(
            self._action_space.n)
        self._generation_kwargs = generation_kwargs
        self._prompt_truncation_side = prompt_truncation_side

    def _build_model_heads(self,
                           model_name: str):
        self._policy_model = AutoModelForCausalLM.from_pretrained(
            model_name)
        self._policy_model.__class__ = override_generation_routines(
            type(self._policy_model))

        self._value_model = AutoModelForCausalLM.from_pretrained(
            model_name)
        self._ref_model = deepcopy(self._policy_model).eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False)

        # apply model parallel
        if torch.cuda.is_available() and self._apply_model_parallel:
            if self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
            if self._value_model.is_parallelizable:
                self._value_model.parallelize()
        self._value_head = self._value_head.to(self.device)

    def _setup_optimizer(self, optimizer_kwargs: Dict[str, Any],
                         weight_decay: float, optimizer_class: torch.optim):
        params = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_kwargs)

    def _prepare_inputs_for_model(self, model: AutoModelForCausalLM,
                                  input_ids: torch.tensor,
                                  model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        if self._apply_model_parallel:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {key: value.to(model.transformer.first_device) if isinstance(
                value, torch.Tensor) else value for key, value in model_inputs.items()}
        return model_inputs

    def get_distribution(self, obs: TensorDict, detach=False):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]
        past_model_kwargs = {
            "attention_mask": attention_mask,
        }

        if detach:
            with torch.no_grad():
                model_inputs = self._prepare_inputs_for_model(self._policy_model,
                                                            input_ids,
                                                            past_model_kwargs)

                # forward pass to transformers
                output = self._policy_model(
                    output_hidden_states=True, **model_inputs)
        else:
                model_inputs = self._prepare_inputs_for_model(self._policy_model,
                                                            input_ids,
                                                            past_model_kwargs)

                # forward pass to transformers
                output = self._policy_model(
                    output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        return dist

    def predict_values(self, obs: TensorDict):
        values, _ = self.forward_value(obs)
        return values
        

    def forward_policy(self, obs: TensorDict,
                       actions: torch.tensor,
                       past_model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            # take attention mask only for the first step
            # for subsequent steps, update_model_kwargs will handle it
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._policy_model,
                                                      input_ids,
                                                      past_model_kwargs)

        # forward pass to transformers
        output = self._policy_model(
            output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        entropy = dist.entropy()

        # sample act
        log_prob = dist.log_prob(actions)

        # update the model kwargs for further generation
        past_model_kwargs = self._policy_model._update_model_kwargs_for_generation(
            output, past_model_kwargs, is_encoder_decoder=self._policy_model.config.is_encoder_decoder
        )

        return actions, log_prob, entropy, output, past_model_kwargs

    def forward_value(self, obs: TensorDict,
                      past_model_kwargs: Optional[Dict[str, torch.tensor]] = None):

        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._value_model,
                                                      input_ids,
                                                      past_model_kwargs)

        # forward pass to transformers
        output = self._value_model(
            output_hidden_states=True, **model_inputs)

        # pool the hidden states ?
        last_tokens_hidden = output.hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        past_model_kwargs = self._value_model._update_model_kwargs_for_generation(
            output, past_model_kwargs, is_encoder_decoder=self._value_model.config.is_encoder_decoder
        )

        return values, past_model_kwargs

    def forward(self, *args, **kwargs):
        # dummy just to comply with base policy
        pass

    @staticmethod
    def _predict(self, observation: Dict[str, torch.tensor],
                 deterministic: bool = False) -> torch.Tensor:
        # dummy just to comply with base policy
        pass

    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _, log_prob, entropy, _, _ = self.forward_policy(obs=obs,
                                                         actions=actions)
        values, _ = self.forward_value(obs)

        return values, log_prob, entropy

    def get_log_probs_ref_model(self, obs: TensorDict,
                                action: torch.tensor,
                                past_model_kwargs: Dict[str, Any] = None):
        self._ref_model = self._ref_model.eval()

        input_ids = obs["input_encoded_pt"]
        attention_mask = obs["input_attention_mask_pt"]

        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._ref_model,
                                                      input_ids,
                                                      past_model_kwargs)
        output = self._ref_model(
            output_hidden_states=True, **model_inputs)
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        log_prob = dist.log_prob(action)

        # update the model kwargs for further generation
        past_model_kwargs = self._ref_model._update_model_kwargs_for_generation(
            output, past_model_kwargs, is_encoder_decoder=self.is_encoder_decoder(self._ref_model)
        )
        return log_prob, past_model_kwargs

    def to(self, device):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)

    def get_policy_first_device(self):
        return self._policy_model.transformer.first_device if self._apply_model_parallel else self._policy_model.device

    def is_encoder_decoder(self, model):
        if isinstance(model, torch.nn.DataParallel):
            return model.module.config.is_encoder_decoder
        else:
            return model.config.is_encoder_decoder


    def generate(self, tokenizer: AutoTokenizer,
                 texts: List[str] = None,
                 max_prompt_length: int = None,
                 input_ids: torch.tensor = None,
                 attention_mask: torch.tensor = None,
                 gen_kwargs: Dict[str, Any] = None):

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self._generation_kwargs

        # switch to eval
        self._policy_model.eval()

        if input_ids is None and\
                attention_mask is None and\
                texts is not None and \
                max_prompt_length is not None:
            # override truncation side for prompt
            prev_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = self._prompt_truncation_side
            encodings = tokenizer(texts,
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
        if "min_length" in gen_kwargs.keys() and not self.is_encoder_decoder(self._policy_model):
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_[
                "min_length"] = input_ids.shape[1] + gen_kwargs["min_length"]
        else:
            generation_kwargs_ = gen_kwargs

        # generate
        gen_output = unwrap_model(self._policy_model).generate(
            inputs=input_ids.to(
                self.get_policy_first_device()),
            attention_mask=attention_mask.to(
                self.get_policy_first_device()),
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs_)

        # number of tokens generated
        seq_length = len(gen_output["scores"])

        # get only the generated text (excluding prompt)
        gen_tokens = gen_output["sequences"][:, -seq_length:]

        # to texts
        gen_texts = [tokenizer.decode(
            output, skip_special_tokens=True)
            for output in gen_tokens.tolist()]

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

        gen_output = {
            "step_wise_logprobs": step_wise_logprobs,
            "step_wise_actions": step_wise_actions,
            "gen_tokens": gen_tokens,
            "gen_texts": gen_texts
        }
        return gen_output

    def get_language_model(self):
        return unwrap_model(self._policy_model)

    def get_inputs_for_generation(self, obs: TensorDict):
        return obs["input_encoded_pt"], obs["input_attention_mask_pt"]

    def get_config_module(self):
        return self._policy_model.transformer

    def get_policy_type(self):
        return PolicyType.CAUSAL


class Seq2SeqLMActorCriticPolicy(LMActorCriticPolicy):
    def _build_model_heads(self,
                           model_name: str):
        self._policy_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name)
        self._policy_model.__class__ = override_generation_routines(
            type(self._policy_model))

        self._value_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name)
        self._ref_model = deepcopy(self._policy_model).eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False)

        # apply model parallel
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
                self._value_model.parallelize()
                self._value_head = self._value_head.to(self.device)
            else: # else defaults to data parallel
                self._policy_model = torch.nn.DataParallel(self._policy_model)
                self._ref_model = torch.nn.DataParallel(self._ref_model)
                self._value_model = torch.nn.DataParallel(self._value_model)
                self._value_head = torch.nn.DataParallel(self._value_head.to(self.device))


    def forward_policy(self, obs: TensorDict,
                       actions: torch.tensor,
                       model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        if model_kwargs is None:
            # 1. prepare model inputs
            model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, model_kwargs = unwrap_model(self._policy_model)._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, model_kwargs)

            # 2. prepare encoder outputs
            model_kwargs = unwrap_model(self._policy_model)._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._policy_model).prepare_inputs_for_generation(input_ids,
                                                                        **model_kwargs)

        # and forward pass to get next token logits
        outputs = self._policy_model(
            **model_inputs,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        # get log probs
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        # update the model kwargs for further generation
        model_kwargs = unwrap_model(self._policy_model)._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=unwrap_model(self._policy_model).config.is_encoder_decoder
        )
        model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)), dim=-1)
        return actions, log_prob, entropy, outputs, model_kwargs

    def forward_value(self, obs: TensorDict,
                      model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        if model_kwargs is None:
            # 1. prepare model inputs
            model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, model_kwargs = unwrap_model(self._value_model)._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, model_kwargs)

            # 2. prepare encoder outputs
            model_kwargs = unwrap_model(self._value_model)._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._value_model).prepare_inputs_for_generation(input_ids,
                                                                       **model_kwargs)

        # and forrward pass to get hidden states
        outputs = self._value_model(
            **model_inputs,
            output_hidden_states=True,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True)

        # get decoder's last hidden state
        last_tokens_hidden = outputs.decoder_hidden_states[-1][:, -1, :].to(
            self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        model_kwargs = unwrap_model(self._value_model)._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=unwrap_model(self._value_model).config.is_encoder_decoder
        )
        model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)), dim=-1)

        return values, model_kwargs

    def get_log_probs_ref_model(self, obs: TensorDict,
                                action: torch.tensor,
                                model_kwargs: Dict[str, Any] = None):
        if model_kwargs is None:
            # 1. prepare model inputs
            model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, model_kwargs = unwrap_model(self._ref_model)._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, model_kwargs)

            # 2. prepare encoder outputs
            model_kwargs = unwrap_model(self._ref_model)._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._ref_model).prepare_inputs_for_generation(input_ids,
                                                                        **model_kwargs)

        # and forward pass to get next token logits
        outputs = self._ref_model(
            **model_inputs,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        # get log probs
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        log_prob = dist.log_prob(action)

        # update the model kwargs for further generation
        model_kwargs = unwrap_model(self._ref_model)._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=unwrap_model(self._ref_model).config.is_encoder_decoder
        )
        model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)), dim=-1)
        return log_prob, model_kwargs

    def get_policy_first_device(self):
        return self._policy_model.get_encoder().first_device if self._apply_model_parallel else self.device

    def get_inputs_for_generation(self, obs: TensorDict):
        return obs["prompt_or_input_encoded_pt"], obs["prompt_or_input_attention_mask_pt"]

    def get_config_module(self):
        return self._policy_model.get_encoder()

    def get_policy_type(self):
        return PolicyType.SEQ2SEQ


class MaskableLMActorCriticPolicy(BasePolicy, MaskableActorCriticWarmStartMixin):
    def __init__(self, observation_space: DictSpace,
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
                 mask_type: str = 'learned_top_k',
                 target_update_iterations: int = 1000,
                 prompt_truncation_side: str = "left",
                 state_dict: Dict[str, Any] = None,
                 min_tokens_to_keep: int = 100
                 ):
        super().__init__(observation_space, action_space)
        self.min_tokens_to_keep = min_tokens_to_keep
        self._action_space = action_space
        self._apply_model_parallel = apply_model_parallel
        self.mask_type = mask_type
        self.top_mask = top_mask if top_mask != -1 else self._action_space.n
        self.target_update_iterations = target_update_iterations
        self._build_model_heads(model_name)
        self._setup_optimizer(optimizer_kwargs, weight_decay, optimizer_class)
        self.load_from_dict(state_dict)
        self._action_dist = MaskableCategoricalDistribution(
            self._action_space.n)
        self._ref_action_dist = CategoricalDistribution(self._action_space.n)
        self._mask_action_dist = CategoricalDistribution(self._action_space.n)
        self._generation_kwargs = generation_kwargs
        self.all_special_ids = None
        self._prompt_truncation_side = prompt_truncation_side

    def _build_model_heads(self,
                           model_name: str):
        self._policy_model = AutoModelForCausalLM.from_pretrained(
            model_name)
        self._policy_model.__class__ = override_generation_routines(
            type(self._policy_model))

        self._value_model = AutoModelForCausalLM.from_pretrained(
            model_name)
        self._ref_model = deepcopy(self._policy_model).eval()
        if 'learned' in self.mask_type:
            self._mask_model = deepcopy(self._policy_model).eval()
        else:
            self._mask_model = self._ref_model.eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False)

        # apply model parallel
        if torch.cuda.is_available() and self._apply_model_parallel:
            if self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
                self._mask_model.parallelize()
            if self._value_model.is_parallelizable:
                self._value_model.parallelize()
        self._value_head = self._value_head.to(self.device)

        self.logits_processor = MaskLogitsProcessorCasualLM(
            self._mask_model, self.action_space, self.top_mask, self._apply_model_parallel, self.get_policy_first_device, self.mask_type, self.min_tokens_to_keep)

    def _setup_optimizer(self, optimizer_kwargs: Dict[str, Any],
                         weight_decay: float, optimizer_class: torch.optim):
        params = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_kwargs)

    def _prepare_inputs_for_model(self, model: AutoModelForCausalLM,
                                  input_ids: torch.tensor,
                                  model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        if self._apply_model_parallel:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {key: value.to(model.transformer.first_device) if isinstance(
                value, torch.Tensor) else value for key, value in model_inputs.items()}
        return model_inputs

    def _get_action_masks(self, input_ids: torch.tensor,
                          attention_mask: torch.tensor) -> torch.tensor:
        action_masks = torch.zeros(
            (input_ids.size(0), self.action_space.n)).to(self.device)
        model_kwargs = {
            "attention_mask": attention_mask,
        }
        model_inputs = self._prepare_inputs_for_model(self._mask_model,
                                                      input_ids,
                                                      model_kwargs)
        output = self._mask_model(
            output_hidden_states=True, **model_inputs)

        next_token_logits = output.logits[:, -1, :]
        ref_distr = self._mask_action_dist.proba_distribution(
            action_logits=next_token_logits)
        next_token_probs = ref_distr.distribution.probs
        _, topk_indices = torch.topk(
            next_token_probs, k=self.top_mask, dim=1, sorted=True)
        action_masks = action_masks.scatter(
            index=topk_indices.long(), dim=1, value=1)

        if self.all_special_ids is not None:
            action_masks = action_masks.scatter(
                index=self.all_special_ids, dim=1, value=1)

        action_masks = action_masks.bool()

        return action_masks

    def forward_policy(self, obs: TensorDict,
                       actions: torch.Tensor,
                       action_masks: torch.Tensor = None,
                       past_model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            # take attention mask only for the first step
            # for subsequent steps, update_model_kwargs will handle it
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._policy_model,
                                                      input_ids,
                                                      past_model_kwargs)

        # forward pass to transformers
        output = self._policy_model(
            output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)

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
            output, past_model_kwargs, is_encoder_decoder=self._policy_model.config.is_encoder_decoder
        )

        return actions, log_prob, entropy, output, action_masks, past_model_kwargs

    def forward_value(self, obs: TensorDict,
                      past_model_kwargs: Optional[Dict[str, torch.tensor]] = None):

        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._value_model,
                                                      input_ids,
                                                      past_model_kwargs)

        # forward pass to transformers
        output = self._value_model(
            output_hidden_states=True, **model_inputs)

        # pool the hidden states ?
        last_tokens_hidden = output.hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        past_model_kwargs = self._value_model._update_model_kwargs_for_generation(
            output, past_model_kwargs, is_encoder_decoder=self._value_model.config.is_encoder_decoder
        )

        return values, past_model_kwargs

    def forward(self, *args, **kwargs):
        # dummy just to comply with base policy
        pass

    @staticmethod
    def _predict(self, observation: Dict[str, torch.tensor],
                 deterministic: bool = False) -> torch.Tensor:
        # dummy just to comply with base policy
        pass

    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         action_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        _, log_prob, entropy, _, _, _ = self.forward_policy(obs=obs,
                                                            actions=actions,
                                                            action_masks=action_masks)
        values, _ = self.forward_value(obs)

        return values, log_prob, entropy

    def get_log_probs_ref_model(self, obs: TensorDict,
                                action: torch.tensor,
                                past_model_kwargs: Dict[str, Any] = None):
        self._ref_model = self._ref_model.eval()

        input_ids = obs["input_encoded_pt"]
        attention_mask = obs["input_attention_mask_pt"]

        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._ref_model,
                                                      input_ids,
                                                      past_model_kwargs)
        output = self._ref_model(
            output_hidden_states=True, **model_inputs)
        next_token_logits = output.logits[:, -1, :]
        dist = self._ref_action_dist.proba_distribution(
            action_logits=next_token_logits)
        log_prob = dist.log_prob(action)

        # update the model kwargs for further generation
        past_model_kwargs = self._ref_model._update_model_kwargs_for_generation(
            output, past_model_kwargs, is_encoder_decoder=self._ref_model.config.is_encoder_decoder
        )
        return log_prob, past_model_kwargs

    def to(self, device):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)

    def get_policy_first_device(self):
        return self._policy_model.transformer.first_device

    def get_inputs_for_generation(self, obs: TensorDict):
        return obs["input_encoded_pt"], obs["input_attention_mask_pt"]

    def generate(self, tokenizer: AutoTokenizer,
                 texts: List[str] = None,
                 max_prompt_length: int = None,
                 input_ids: torch.tensor = None,
                 attention_mask: torch.tensor = None,
                 gen_kwargs: Dict[str, Any] = None):

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self._generation_kwargs

        # switch to eval
        self._policy_model.eval()
        self.logits_processor.reset()

        # action_masks = self._get_action_masks(input_ids, attention_mask)
        # curr_logits_processor = [MaskLogitsProcessor(None)]

        if input_ids is None and\
                attention_mask is None and\
                texts is not None and \
                max_prompt_length is not None:
            prev_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = self._prompt_truncation_side
            encodings = tokenizer(texts,
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
            self.get_policy_first_device())
        self.logits_processor.all_special_ids = self.all_special_ids = torch.tensor(
            tokenizer.all_special_ids, dtype=input_ids.dtype, device=self.get_policy_first_device()).unsqueeze(0).expand((input_ids.size(0), -1))

        # if min_length argument is set and if policy is not a seq2seq LM (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if "min_length" in gen_kwargs.keys() and not self._policy_model.config.is_encoder_decoder:
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_[
                "min_length"] = input_ids.shape[1] + gen_kwargs["min_length"]
        else:
            generation_kwargs_ = gen_kwargs

        # generate
        gen_output = self._policy_model.generate(
            inputs=input_ids.to(
                self.get_policy_first_device()),
            attention_mask=attention_mask.to(
                self.get_policy_first_device()),
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=[self.logits_processor],
            **generation_kwargs_)

        # number of tokens generated
        seq_length = len(gen_output["scores"])

        # get only the generated text (excluding prompt)
        gen_tokens = gen_output["sequences"][:, -seq_length:]

        # to texts
        gen_texts = [tokenizer.decode(
            output, skip_special_tokens=True)
            for output in gen_tokens.tolist()]

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

        gen_output = {
            "step_wise_logprobs": step_wise_logprobs,
            "step_wise_actions": step_wise_actions,
            "gen_tokens": gen_tokens,
            "gen_texts": gen_texts,
            "action_masks": action_masks
        }
        return gen_output

    def get_language_model(self):
        return self._policy_model

    def update_mask_model(self):
        self._mask_model = deepcopy(self._policy_model).eval()

    def get_config_module(self):
        return self._policy_model.transformer

    def get_policy_type(self):
        return PolicyType.CAUSAL


class MaskableSeq2SeqLMActorCriticPolicy(MaskableLMActorCriticPolicy):
    def _build_model_heads(self,
                           model_name: str):
        self._policy_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name)
        self._policy_model.__class__ = override_generation_routines(
            type(self._policy_model))

        self._value_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name)
        self._ref_model = deepcopy(self._policy_model).eval()

        if 'learned' in self.mask_type:
            self._mask_model = deepcopy(self._policy_model).eval()
        else:
            self._mask_model = self._ref_model.eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False)

        # apply model parallel
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
                self._mask_model.parallelize()
                self._value_model.parallelize()
                self._value_head = self._value_head.to(self.device)
            else: # else defaults to data parallel
                self._policy_model = torch.nn.DataParallel(self._policy_model)
                self._ref_model = torch.nn.DataParallel(self._ref_model)
                self._mask_model = torch.nn.DataParallel(self._mask_model)
                self._value_model = torch.nn.DataParallel(self._value_model)
                self._value_head = torch.nn.DataParallel(self._value_head.to(self.device))

        self.logits_processor = MaskLogitsProcessorSeq2SeqLM(
            self._mask_model, self.action_space, self.top_mask, self._apply_model_parallel, self.get_policy_first_device, self.mask_type, self.min_tokens_to_keep)

    def _prepare_inputs_for_model(self, model: AutoModelForCausalLM,
                                  input_ids: torch.tensor,
                                  model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        return model_inputs

    def _get_action_masks(self, model_inputs, decoder_attn_mask) -> torch.tensor:
        action_masks = torch.zeros(
            (decoder_attn_mask.size(0), self.action_space.n)).to(self.device)
        outputs = self._mask_model(
            **model_inputs,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        ref_distr = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        next_token_probs = ref_distr.distribution.probs
        _, topk_indices = torch.topk(
            next_token_probs, k=self.top_mask, dim=1, sorted=True)
        action_masks = action_masks.scatter(
            index=topk_indices.long(), dim=1, value=1)

        if self.all_special_ids is not None:
            action_masks = action_masks.scatter(
                index=self.all_special_ids, dim=1, value=1)
        action_masks = action_masks.bool()

        return action_masks

    def forward_policy(self, obs: TensorDict,
                       actions: torch.Tensor,
                       action_masks: torch.Tensor = None,
                       model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        if model_kwargs is None:
            # 1. prepare model inputs
            model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, model_kwargs = self._policy_model._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, model_kwargs)

            # 2. prepare encoder outputs
            model_kwargs = self._policy_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._policy_model).prepare_inputs_for_generation(input_ids,
                                                                        **model_kwargs)
        # and forward pass to get next token logits
        outputs = self._policy_model(
            **model_inputs,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        if action_masks is None:
            action_masks = self._get_action_masks(
                model_inputs, decoder_attn_mask)

        # get log probs
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        raw_log_probs = dist.log_prob(actions)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # update the model kwargs for further generation
        model_kwargs = unwrap_model(self._policy_model)._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=unwrap_model(self._policy_model).config.is_encoder_decoder
        )
        model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)), dim=-1)
        return actions, raw_log_probs, log_probs, entropy, outputs, action_masks, model_kwargs

    def forward_value(self, obs: TensorDict,
                      model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        if model_kwargs is None:
            # 1. prepare model inputs
            model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, model_kwargs = self._value_model._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, model_kwargs)

            # 2. prepare encoder outputs
            model_kwargs = self._value_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._value_model).prepare_inputs_for_generation(input_ids,
                                                                       **model_kwargs)

        # and forrward pass to get hidden states
        outputs = self._value_model(
            **model_inputs,
            output_hidden_states=True,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True)

        # get decoder's last hidden state
        last_tokens_hidden = outputs.decoder_hidden_states[-1][:, -1, :].to(
            self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        model_kwargs = unwrap_model(self._value_model)._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=unwrap_model(self._value_model).config.is_encoder_decoder
        )
        model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)), dim=-1)

        return values, model_kwargs

    def get_log_probs_ref_model(self, obs: TensorDict,
                                action: torch.tensor,
                                model_kwargs: Dict[str, Any] = None):
        if model_kwargs is None:
            # 1. prepare model inputs
            model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, model_kwargs = self._ref_model._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, model_kwargs)

            # 2. prepare encoder outputs
            model_kwargs = self._ref_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._ref_model).prepare_inputs_for_generation(input_ids,
                                                                        **model_kwargs)

        # and forward pass to get next token logits
        outputs = self._ref_model(
            **model_inputs,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        # get log probs
        dist = self._ref_action_dist.proba_distribution(
            action_logits=next_token_logits)
        log_prob = dist.log_prob(action)

        # update the model kwargs for further generation
        model_kwargs = unwrap_model(self._ref_model)._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=unwrap_model(self._ref_model).config.is_encoder_decoder
        )
        model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)), dim=-1)
        return log_prob, model_kwargs

    def get_policy_first_device(self):
        return self._policy_model.get_encoder().first_device

    def get_inputs_for_generation(self, obs: TensorDict):
        return obs["prompt_or_input_encoded_pt"], obs["prompt_or_input_attention_mask_pt"]

    def get_config_module(self):
        return self._policy_model.get_encoder()

    def get_policy_type(self):
        return PolicyType.SEQ2SEQ