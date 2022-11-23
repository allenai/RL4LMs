from typing import Dict, Optional, Union

import torch
from torch.distributions.utils import logits_to_probs, probs_to_logits
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, LogitsProcessor
from transformers.modeling_utils import unwrap_model

from rl4lms.algorithms.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)


class MaskLogitsProcessorCasualLM(LogitsProcessor):
    def __init__(
        self,
        mask_model,
        action_space,
        top_mask,
        apply_model_parallel,
        get_policy_first_device,
        mask_type,
        min_tokens_to_keep,
    ):
        super(MaskLogitsProcessorCasualLM, self).__init__()
        self.mask_model = mask_model
        self.past_model_kwargs = None
        self.attention_mask = None
        self.action_masks = None
        self.action_space = action_space
        self.top_mask = top_mask
        self._apply_model_parallel = apply_model_parallel
        self._action_dist = MaskableCategoricalDistribution(self.action_space.n)
        self.get_policy_first_device = get_policy_first_device
        self.mask_type = mask_type
        self.min_tokens_to_keep = min_tokens_to_keep

    def reset(self):
        self.past_model_kwargs = None
        self.attention_mask = None
        self.action_masks = None
        self.all_special_ids = None
        # self._action_dist = MaskableCategoricalDistribution(
        #     self.action_space.n)

    def _prepare_inputs_for_model(
        self,
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        input_ids: torch.Tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        model_inputs = unwrap_model(model).prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )

        if self._apply_model_parallel:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {
                key: value.to(self.get_policy_first_device())
                if isinstance(value, torch.Tensor)
                else value
                for key, value in model_inputs.items()
            }
        return model_inputs

    def _get_action_masks(
        self,
        input_ids: torch.Tensor,
        scores: torch.FloatTensor,
        model_inputs: dict = None,
    ) -> torch.Tensor:
        action_masks = torch.zeros((input_ids.size(0), self.action_space.n)).to(
            self.device
        )
        self.attention_mask = model_inputs["attention_mask"]
        self.past_model_kwargs = {"attention_mask": self.attention_mask}
        if model_inputs is not None:
            model_inputs = self._prepare_inputs_for_model(
                self.mask_model, input_ids, self.past_model_kwargs
            )
        if self.mask_type == "learned_top_k":
            output = self.mask_model(output_hidden_states=True, **model_inputs)

            next_token_logits = output.logits[:, -1, :]
            ref_distr = self._action_dist.proba_distribution(
                action_logits=next_token_logits
            )
            next_token_probs = ref_distr.distribution.probs
            # assert torch.sum(action_masks.long()).item() != 0
            self.past_model_kwargs = (
                self.mask_model._update_model_kwargs_for_generation(
                    output,
                    self.past_model_kwargs,
                    is_encoder_decoder=self.mask_model.config.is_encoder_decoder,
                )
            )
            _, indices_to_remove = torch.topk(
                next_token_probs, k=self.top_mask, dim=1, sorted=True
            )
            action_masks = action_masks.scatter(
                index=indices_to_remove.long(), dim=1, value=1
            )
        elif self.mask_type == "learned_top_p":
            output = self.mask_model(output_hidden_states=True, **model_inputs)

            next_token_logits = output.logits[:, -1, :]
            # ref_distr = self._action_dist.proba_distribution(
            #     action_logits=next_token_logits)
            # next_token_logits = ref_distr.distribution.logits

            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_mask
            if self.min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            action_masks = (~indices_to_remove).long()
        elif self.mask_type == "topk":
            next_token_probs = scores
            _, indices_to_remove = torch.topk(
                next_token_probs, k=self.top_mask, dim=1, sorted=True
            )
            action_masks = action_masks.scatter(
                index=indices_to_remove.long(), dim=1, value=1
            )
        else:
            raise NotImplementedError

        # always unmask the special ids like bos and eos
        action_masks = action_masks.scatter(index=self.all_special_ids, dim=1, value=1)
        action_masks = action_masks.bool()

        return action_masks

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        model_inputs: dict = None,
    ) -> torch.FloatTensor:
        self.device = scores.device
        action_masks = self._get_action_masks(input_ids, scores, model_inputs)
        action_masks = torch.as_tensor(
            action_masks, dtype=torch.bool, device=self.device
        ).reshape(scores.shape)
        scores = torch.masked_fill(scores, ~action_masks, -float("Inf"))
        return scores


class MaskLogitsProcessorSeq2SeqLM(LogitsProcessor):
    def __init__(
        self,
        mask_model,
        action_space,
        top_mask,
        apply_model_parallel,
        get_policy_first_device,
        mask_type,
        min_tokens_to_keep,
    ):
        super(MaskLogitsProcessorSeq2SeqLM, self).__init__()
        self.mask_model = mask_model
        self.past_model_kwargs = None
        self.attention_mask = None
        self.action_masks = None
        self.action_space = action_space
        self.top_mask = top_mask
        self._apply_model_parallel = apply_model_parallel
        self._action_dist = MaskableCategoricalDistribution(self.action_space.n)
        self.get_policy_first_device = get_policy_first_device
        self.mask_type = mask_type
        self.min_tokens_to_keep = min_tokens_to_keep

    def reset(self):
        self.past_model_kwargs = None
        self.attention_mask = None
        self.action_masks = None
        self.all_special_ids = None
        # self._action_dist = MaskableCategoricalDistribution(
        #     self.action_space.n)

    def _prepare_inputs_for_model(
        self,
        model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        input_ids: torch.Tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if self._apply_model_parallel:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {
                key: value.to(self.get_policy_first_device())
                if isinstance(value, torch.Tensor)
                else value
                for key, value in model_inputs.items()
            }
        return model_inputs

    def _get_action_masks(
        self,
        input_ids: torch.Tensor,
        scores: torch.FloatTensor,
        model_inputs: dict = None,
    ) -> torch.Tensor:
        action_masks = torch.zeros((input_ids.size(0), self.action_space.n)).to(
            self.device
        )

        self.past_model_kwargs = model_inputs

        # and forward pass to get next token logits
        if self.mask_type == "learned_top_k":
            output = self.mask_model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )

            next_token_logits = output.logits[:, -1, :]
            ref_distr = self._action_dist.proba_distribution(
                action_logits=next_token_logits
            )
            next_token_probs = ref_distr.distribution.probs
            # assert torch.sum(action_masks.long()).item() != 0
            self.past_model_kwargs = (
                self.mask_model._update_model_kwargs_for_generation(
                    output,
                    self.past_model_kwargs,
                    is_encoder_decoder=self.mask_model.config.is_encoder_decoder,
                )
            )
            _, indices_to_remove = torch.topk(
                next_token_probs, k=self.top_mask, dim=1, sorted=True
            )
            action_masks = action_masks.scatter(
                index=indices_to_remove.long(), dim=1, value=1
            )
        elif self.mask_type == "learned_top_p":
            output = self.mask_model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )

            next_token_logits = output.logits[:, -1, :]
            # ref_distr = self._action_dist.proba_distribution(
            #     action_logits=next_token_logits)
            # next_token_logits = ref_distr.distribution.logits

            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_mask
            if self.min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            action_masks = (~indices_to_remove).long()
        elif self.mask_type == "topk":
            next_token_probs = scores
            _, indices_to_remove = torch.topk(
                next_token_probs, k=self.top_mask, dim=1, sorted=True
            )
            action_masks = action_masks.scatter(
                index=indices_to_remove.long(), dim=1, value=1
            )
        else:
            raise NotImplementedError

        # always unmask the special ids like bos and eos
        action_masks = action_masks.scatter(index=self.all_special_ids, dim=1, value=1)
        action_masks = action_masks.bool()

        return action_masks

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        model_inputs: dict = None,
    ) -> torch.FloatTensor:
        self.device = scores.device
        action_masks = self._get_action_masks(input_ids, scores, model_inputs)
        action_masks = torch.as_tensor(
            action_masks, dtype=torch.bool, device=self.device
        ).reshape(scores.shape)
        scores = torch.masked_fill(scores, ~action_masks, -float("Inf"))
        return scores
