from transformers import AutoTokenizer, PreTrainedModel, TrainerCallback, PreTrainedTokenizer

from rl4lms.data_pools.custom_text_generation_pools import Sample
from rl4lms.data_pools.text_generation_pool import TextGenPool
from rl4lms.envs.text_generation.logging_utils import Tracker
from typing import List, Dict, Any
from copy import deepcopy
from tqdm import tqdm
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader
from rl4lms.envs.text_generation.registry import PostProcessorRegistry, MetricRegistry
from accelerate import Accelerator
from rl4lms.envs.text_generation.metric import BaseMetric
from rl4lms.envs.text_generation.policy.base_policy import GenerationOutputs
from transformers.modeling_utils import unwrap_model
import torch

def get_batch(samples: List[Sample], batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix:current_ix + batch_size]
        yield current_batch
        current_ix += batch_size


def evaluate_on_samples(model: PreTrainedModel,
                        tokenizer: AutoTokenizer,
                        dataloader: DataLoader,
                        max_prompt_length: int,
                        metrics: List[BaseMetric],
                        epoch: int,
                        split_name: str,
                        tracker: Tracker = None,
                        generation_kwargs: Dict[str, Any] = {},
                        accelerator: Accelerator = None
                        ):
    

    # tracker
    tracker.log_info("DISTRIBUTED EVALUATION STARTED")

    # wait for everyone
    accelerator.wait_for_everyone()

    # generate text by batch
    generations_by_sample_ids = {}
    for batch in tqdm(dataloader, desc="DIST EVALUATION", disable=not accelerator.is_local_main_process):
        batch_sample_ids, batch_generated_texts = generate_text(
            model, tokenizer, batch, accelerator, max_prompt_length, generation_kwargs
        )

        for sample_id, gen_text in zip(batch_sample_ids, batch_generated_texts):
            generations_by_sample_ids[sample_id] = gen_text

    # tracker
    tracker.log_info("DISTRIBUTED EVALUATION FINISHED")

    if accelerator.is_main_process:
        # compute metrics
        sample_predictions_dict, corpus_level_metrics = compute_metrics(dataloader, 
                                                                        metrics, 
                                                                        generations_by_sample_ids, 
                                                                        split_name, 
                                                                        model,
                                                                        accelerator)

        if tracker is not None:
            # log the entire predictions
            tracker.log_predictions(epoch, split_name, sample_predictions_dict)
            # log the corpus level scores
            tracker.log_metrics(epoch, split_name, corpus_level_metrics)


def generate_text(model: PreTrainedModel,
                  tokenizer: AutoTokenizer,
                  samples: List[Sample],
                  accelerator: Accelerator,
                  max_prompt_length: int,
                  generation_kwargs: Dict[str, Any]
                  ):
    prompt_texts = [
        sample.prompt_or_input_text for sample in samples
    ]
    sample_ids = torch.tensor([sample.id for sample in samples]).to(accelerator.device)
    generated_output =generate(accelerator.unwrap_model(model), 
        tokenizer, accelerator, prompt_texts, sample_ids, max_prompt_length, generation_kwargs
    )
    return generated_output.sample_ids, generated_output.gen_texts


def generate(model: PreTrainedModel,
             tokenizer: AutoTokenizer,
             accelerator: Accelerator,
             texts: List[str] = None,
             sample_ids: torch.tensor = None,
             max_prompt_length: int = None,
             generation_kwargs: Dict[str, Any] = {}):

    # switch to eval
    model.eval()

    encodings = tokenizer(texts,
                          padding="max_length",
                          max_length=max_prompt_length,
                          return_tensors="pt",
                          return_attention_mask=True,
                          truncation=True,
                          )
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    # if min_length argument is set and if policy is not a seq2seq LM (ie. causal LM)
    # then it has to be adjusted to input_size + min_length
    if generation_kwargs.get("min_length", None) is not None and not unwrap_model(model).config.is_encoder_decoder:
        generation_kwargs_ = deepcopy(generation_kwargs)
        generation_kwargs_[
            "min_length"] = input_ids.shape[1] + generation_kwargs["min_length"]
    else:
        generation_kwargs_ = generation_kwargs

    # generate
    gen_output = accelerator.unwrap_model(model).generate(
        inputs=input_ids.to(accelerator.device),
        attention_mask=attention_mask.to(accelerator.device),
        return_dict_in_generate=True,
        output_scores=True,
        **generation_kwargs_)

    # number of tokens generated
    seq_length = len(gen_output["scores"])

    # get only the generated text (excluding prompt)
    gen_tokens = gen_output["sequences"][:, -seq_length:]

    # now we have to gather from all devices
    # first pad the gen_tokens to maximum sequence length
    max_length = generation_kwargs_["max_new_tokens"]  # TBD: fix this
    padded_gen_tokens = torch.ones((gen_tokens.shape[0], max_length), dtype=torch.int32).to(accelerator.device) * tokenizer.eos_token_id
    padded_gen_tokens[:,:seq_length] = gen_tokens
    gathered_gen_tokens = accelerator.gather_for_metrics(padded_gen_tokens)

    gathered_gen_texts = []
    for output in gathered_gen_tokens.tolist():
        text = tokenizer.decode(output, skip_special_tokens=True)
        gathered_gen_texts.append(text)

    gathered_sample_ids = accelerator.gather_for_metrics(sample_ids).tolist()
    assert len(gathered_gen_texts) == len(gathered_sample_ids)

    generation_outputs = GenerationOutputs(None, None, None, gathered_gen_texts, None, gathered_sample_ids)
    return generation_outputs


class EvalCallack(TrainerCallback):
    def __init__(self, val_dataloader: DataLoader,
                 generation_kwargs: Dict[str, Any],
                 eval_batch_size: int,
                 tokenizer: PreTrainedTokenizer,
                 metrics: List[BaseMetric],
                 max_prompt_length: int,
                 tracker: Tracker,
                 accelerator: Accelerator):
        self._val_dataloader = val_dataloader
        self._gen_kwargs = generation_kwargs
        self._tokenizer = tokenizer
        self._metrics = metrics
        self._eval_batch_size = eval_batch_size
        self._max_prompt_length = max_prompt_length
        self._tracker = tracker
        self._accelerator = accelerator

    def on_log(self, args, state, control, logs=None, **kwargs):
        print("Evaluation")
        model = kwargs.pop("model")
        #model = self._accelerator.prepare(model)
        batch = self._tokenizer(["random text"], return_tensors="pt").to(self._accelerator.device)
        outputs = model(**batch)
        evaluate_on_samples(model,
                            self._tokenizer,
                            self._val_dataloader,
                            self._max_prompt_length,
                            self._metrics,
                            state.epoch,
                            "val",
                            tracker=self._tracker,
                            generation_kwargs=self._gen_kwargs,
                            accelerator=self._accelerator)


def get_datasets_for_causal(train_datapool: TextGenPool):
    texts = []
    for sample, _ in train_datapool:
        for ref in sample.references:
            text = sample.prompt_or_input_text + ref
            texts.append(text)

    train_dataset = Dataset.from_dict(
        {
            "content": texts
        },
        split="train"
    )
    return train_dataset


def get_datasets_for_seq2seq(train_datapool: TextGenPool):
    articles = []
    summaries = []
    for sample, _ in train_datapool:
        for ref in sample.references:
            articles.append(sample.prompt_or_input_text)
            summaries.append(ref)

    train_dataset = Dataset.from_dict(
        {
            "input_text": articles,
            "output_text": summaries
        },
        split="train"
    )
    return train_dataset


def tokenize_causal(item, tokenizer):
    outputs = tokenizer(
        item["content"],
        truncation=True,
    )
    return {"input_ids": outputs["input_ids"]}


def tokenize_seq2seq(item, tokenizer):
    model_inputs = tokenizer(
        item["input_text"],
        truncation=True,

    )
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            item["output_text"],
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(dataloader: DataLoader,
                    metrics: List[BaseMetric],
                    generations_by_sample_ids: Dict[int, str],
                    split_name: str,
                    model: PreTrainedModel,
                    accelerator: Accelerator):

    # gather everything
    all_ref_texts = []
    all_prompt_texts = []
    all_meta_infos = []
    all_generated_texts = []
    all_sample_ids = []
    for sample in dataloader.dataset:
        all_ref_texts.append(sample.references)
        all_prompt_texts.append(sample.prompt_or_input_text)
        all_generated_texts.append(generations_by_sample_ids[sample.id])
        all_meta_infos.append(sample.meta_data)
        all_sample_ids.append(sample.id)


    # gather metrics
    corpus_level_metrics = {}
    sample_scores_by_metric = {}
    n_samples = len(all_sample_ids)
    if metrics is not None:
        for metric in metrics:
            metric_dict = metric.compute(
                all_prompt_texts,
                all_generated_texts,
                all_ref_texts,
                all_meta_infos,
                accelerator.unwrap_model(model),
                split_name,
            )

            for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                if sample_scores is None:
                    sample_scores = ["n/a"] * n_samples
                corpus_level_metrics[metric_key] = corpus_score
                sample_scores_by_metric[metric_key] = sample_scores

    # aggregate sample metric scores
    sample_predictions_dict = []
    for ix, (sample_id, prompt_text, generated_text, ref_texts) in enumerate(
        zip(all_sample_ids, all_prompt_texts, all_generated_texts, all_ref_texts)
    ):
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample_id,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "ref_text": "".join(
                [
                    f"<START-{ref_ix+1}>" + ref_text + f"<END-{ref_ix+1}>"
                    for ref_ix, ref_text in enumerate(ref_texts)
                ]
            ),
        }
        for metric_key, sample_scores in sample_scores_by_metric.items():
            sample_prediction[metric_key] = sample_scores[ix]
        sample_predictions_dict.append(sample_prediction)


    return sample_predictions_dict, corpus_level_metrics


def generate_on_samples(model: PreTrainedModel,
                        tokenizer: AutoTokenizer,
                        samples: List[Sample],
                        batch_size: int,
                        max_prompt_length: int,
                        generation_kwargs: Dict[str, Any] = {},
                        ):
    # post-processing fn
    generation_kwargs = deepcopy(generation_kwargs)
    post_processing_fn = generation_kwargs.pop("post_processing_fn")
    if post_processing_fn is not None:
        post_processing_fn = PostProcessorRegistry.get(
            post_processing_fn["id"])

    # generate text by batch
    all_generated_texts = []
    all_ref_texts = []
    all_prompt_texts = []
    all_meta_infos = []
    for batch in tqdm(list(get_batch(samples, batch_size)), desc="Predicting"):
        batch_generated_texts = generate_text(
            model, tokenizer, batch, max_prompt_length, generation_kwargs)

        # post-processing of generated text
        if post_processing_fn is not None:
            batch_generated_texts = [post_processing_fn(
                text) for text in batch_generated_texts]

        batch_ref_texts = [sample.references for sample in batch]
        batch_prompt_texts = [sample.prompt_or_input_text for sample in batch]
        batch_meta_infos = [sample.meta_data for sample in batch]
        all_generated_texts.extend(batch_generated_texts)
        all_ref_texts.extend(batch_ref_texts)
        all_prompt_texts.extend(batch_prompt_texts)
        all_meta_infos.extend(batch_meta_infos)

    return all_prompt_texts, all_generated_texts, all_ref_texts, all_meta_infos
