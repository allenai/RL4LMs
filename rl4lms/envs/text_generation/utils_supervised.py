from transformers import AutoTokenizer, PreTrainedModel, TrainerCallback, PreTrainedTokenizer

from rl4lms.data_pools.custom_text_generation_pools import Sample
from rl4lms.data_pools.text_generation_pool import TextGenPool
from rl4lms.envs.text_generation.logging_utils import Tracker
from typing import List, Dict, Any
from copy import deepcopy
from tqdm import tqdm
from datasets.arrow_dataset import Dataset
from rl4lms.envs.text_generation.registry import PostProcessorRegistry, MetricRegistry


def get_batch(samples: List[Sample], batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix:current_ix + batch_size]
        yield current_batch
        current_ix += batch_size


def evaluate_on_samples(model: PreTrainedModel,
                        tokenizer: AutoTokenizer,
                        samples: List[Sample],
                        batch_size: int,
                        max_prompt_length: int,
                        metrics_config_dict: Dict[str, Any],
                        epoch: int,
                        split_name: str,
                        tracker: Tracker = None,
                        generation_kwargs: Dict[str, Any] = {},
                        ):
    all_prompt_texts, all_generated_texts, all_ref_texts, all_meta_infos = generate_on_samples(
        model, tokenizer, samples, batch_size, max_prompt_length, generation_kwargs)

    corpus_level_metrics, sample_predictions_dict = compute_metrics(
        metrics_config_dict, samples, all_prompt_texts, all_generated_texts, all_ref_texts, all_meta_infos, split_name, model)

    if tracker is not None:
        # log the entire predictions
        tracker.log_predictions(epoch, split_name, sample_predictions_dict)
        # log the corpus level scores
        tracker.log_metrics(epoch, split_name, corpus_level_metrics)

    return corpus_level_metrics


def generate_text(model: PreTrainedModel,
                  tokenizer: AutoTokenizer,
                  samples: List[Sample],
                  max_prompt_length: int,
                  generation_kwargs
                  ):
    prompt_texts = [sample.prompt_or_input_text for sample in samples]
    generated_texts = generate(model,
                               tokenizer,
                               prompt_texts,
                               max_prompt_length,
                               generation_kwargs)
    return generated_texts


def generate(model: PreTrainedModel,
             tokenizer: AutoTokenizer,
             texts: List[str] = None,
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
    if generation_kwargs.get("min_length", None) is not None and not model.config.is_encoder_decoder:
        generation_kwargs_ = deepcopy(generation_kwargs)
        generation_kwargs_[
            "min_length"] = input_ids.shape[1] + generation_kwargs["min_length"]
    else:
        generation_kwargs_ = generation_kwargs

    if model.config.is_encoder_decoder:
        # seq2seq LM
        first_device = model.encoder.first_device
    else:
        # causal LM
        first_device = model.transformer.first_device

    # generate
    gen_output = model.generate(
        inputs=input_ids.to(first_device),
        attention_mask=attention_mask.to(first_device),
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

    return gen_texts


class EvalCallack(TrainerCallback):
    def __init__(self, val_samples: List[Sample],
                 generation_kwargs: Dict[str, Any],
                 eval_batch_size: int,
                 tokenizer: PreTrainedTokenizer,
                 metrics_config_dict: Dict[str, Any],
                 max_prompt_length: int,
                 tracker: Tracker):
        self._val_samples = val_samples
        self._gen_kwargs = generation_kwargs
        self._tokenizer = tokenizer
        self._metrics_config_dict = metrics_config_dict
        self._eval_batch_size = eval_batch_size
        self._max_prompt_length = max_prompt_length
        self._tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.pop("model")
        evaluate_on_samples(model,
                            self._tokenizer,
                            self._val_samples,
                            self._eval_batch_size,
                            self._max_prompt_length,
                            self._metrics_config_dict,
                            state.epoch,
                            "val",
                            tracker=self._tracker,
                            generation_kwargs=self._gen_kwargs)


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


def compute_metrics(metrics_config_dict: List[Dict[str, Any]],
                    samples: List[Sample],
                    all_prompt_texts: List[str],
                    all_generated_texts: List[str],
                    all_ref_texts: List[str],
                    all_meta_infos: List[Dict[str, Any]],
                    split_name: str,
                    model: PreTrainedModel):
    # compute metrics
    n_samples = len(samples)
    corpus_level_metrics = {}
    sample_scores_by_metric = {}
    if metrics_config_dict is not None:
        for metric_config in tqdm(metrics_config_dict, desc="Computing metrics"):
            # instantiate the config here
            metric = MetricRegistry.get(
                metric_config["id"], metric_config.get("args", {}))
            metric_dict = metric.compute(
                all_prompt_texts, all_generated_texts, all_ref_texts, all_meta_infos, model, split_name)

            for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                if sample_scores is None:
                    sample_scores = ["n/a"] * n_samples
                corpus_level_metrics[metric_key] = corpus_score
                sample_scores_by_metric[metric_key] = sample_scores

    # aggregate sample metric scores
    sample_predictions_dict = []
    for ix, (sample, prompt_text, generated_text, ref_texts) in enumerate(zip(samples,
                                                                              all_prompt_texts,
                                                                              all_generated_texts,
                                                                              all_ref_texts)):
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample.id,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "ref_text": "".join([f"<START-{ref_ix+1}>"+ref_text+f"<END-{ref_ix+1}>"
                                 for ref_ix, ref_text in enumerate(ref_texts)]),
        }
        for metric_key, sample_scores in sample_scores_by_metric.items():
            sample_prediction[metric_key] = sample_scores[ix]
        sample_predictions_dict.append(sample_prediction)

    return corpus_level_metrics, sample_predictions_dict


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
