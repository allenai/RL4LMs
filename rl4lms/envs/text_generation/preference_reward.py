"""
A (hopefully) Simple API for scoring commongen outputs.


{"input": "pyramids in the desert with a blue sky", "target": "bad"}
{"input": "A huge crane unloads material on the concrete foundation.", "target": "bad"}
{"input": "a ceile overlooks a window on the ground floor.", "target": "good"}
{"input": "a wildfire threatens a home as it burns on the road.", "target": "good"}
{"input": "A man drives a car during a race at an event.", "target": "bad"}

"""

import argparse
import torch
import transformers
import os
import tqdm
import numpy as np
from urllib.request import urlretrieve
from pathlib import Path
from rl4lms.envs.text_generation.reward import BatchedRewardFunction
from typing import List, Dict, Any
from rl4lms.envs.text_generation.metric import MeteorMetric

_model, _tokenizer = None, None

model2url = {
    "11b": "https://storage.googleapis.com/ai2-jack-public/rl4lms_preference_models/t5-11b~commongen_prefs.pt",
}


def get_model(model_type, device=None):
    global _model, model2url
    if model_type not in {
        "11b",
    }:
        raise NotImplementedError(
            '{} is not a valid model please use "11b"'.format(model_type)
        )

    if _model is None:
        hf_model_name = "t5-" + model_type
        print("Loading model: this will run only once.")
        if model_type == "11b":
            model_path = "t5-11b~commongen_prefs.pt"

        # destination path
        dest_base_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "common_gen_preference"
        )
        dest_model_path = os.path.join(dest_base_path, model_path)

        if not os.path.exists(dest_model_path):
            os.makedirs(dest_base_path, exist_ok=True)
            urlretrieve(model2url[model_type], dest_model_path)

        state = torch.load(dest_model_path)
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        _model = transformers.AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
        if (
            model_type == "11b"
        ):  # need to resize due to deepspeed, these entires are not accessed.
            _model.resize_token_embeddings(
                len(transformers.AutoTokenizer.from_pretrained(hf_model_name))
            )
        _model.load_state_dict(state)
        _model.eval()
        if device is not None:
            _model = _model.to(device)

    return _model


def get_tokenizer(model_type):
    global _tokenizer
    if model_type not in {"11b"}:
        raise NotImplementedError(
            '{} is not a valid model please use "11b"'.format(model_type)
        )

    if _tokenizer is None:
        hf_model_name = "t5-" + model_type
        _tokenizer = transformers.T5TokenizerFast.from_pretrained(hf_model_name)

    return _tokenizer


class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        res = self.tokenizer(self.data[idx]["input"], truncation=True)
        res["labels"] = self.tokenizer(self.data[idx]["label"]).input_ids
        return res

    def __len__(self):
        return len(self.data)


def get_scores(inputs, model_type, device=None, batch_size=1, verbose=False):
    """
    Inputs:
      - a list of commongens to score, e.g.,:
      - device: which torch device to load model on, e.g., "cuda:3"
    Outputs:
      - P(good commongen); higher is better
    """
    assert model_type in {"11b"}

    if isinstance(inputs, str):
        inputs = [inputs]

    model = get_model(model_type, device=device)
    tokenizer = get_tokenizer(model_type)

    score_itr = T5Dataset(
        [{"input": inp, "label": "x"} for inp in inputs], tokenizer
    )  # dummy labels for inference
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, return_tensors="pt"
    )
    score_itr = torch.utils.data.DataLoader(
        score_itr, shuffle=False, collate_fn=data_collator, batch_size=batch_size
    )
    score_itr = score_itr if not verbose else tqdm.tqdm(score_itr, total=len(score_itr))

    good_idx, bad_idx = tokenizer("good").input_ids[0], tokenizer("bad").input_ids[0]
    scores = []
    with torch.no_grad():
        for batch in score_itr:
            if device is not None:
                input_ids, attention_mask, targets = (
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                )
            model_output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=targets
            )
            logits_pos = model_output["logits"][:, 0, good_idx].cpu().numpy()
            logits_neg = model_output["logits"][:, 0, bad_idx].cpu().numpy()
            exp_logit_pos, exp_logit_neg = np.exp(logits_pos), np.exp(logits_neg)
            scores.extend(
                list(
                    [float(x) for x in exp_logit_pos / (exp_logit_pos + exp_logit_neg)]
                )
            )
    return scores


def parse_args():
    """
    Optional args for main function, mostly just to test.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="11b", choices={"11b"})
    parser.add_argument("--batch_size", default=1, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(1)

    scores = get_scores(
        [
            "A man drives a car during a race at an event.",
            "A car drives a man during a race at an event.",
            "The event is a man driving a car race.",
        ],
        args.model_type,
        batch_size=args.batch_size,
        device="cpu",
        verbose=False,
    )
    print(scores)

    # t5-11b:
    # [0.7316965460777283, 0.5394992232322693, 0.22872506082057953]


class CommonGenPrefRM(BatchedRewardFunction):
    def __init__(
        self,
        model_type: str,
        device: str,
        batch_size: int,
        concept_penalty_coeff: float = 0.0,
        meteor_coeff: float = 0.0,
    ) -> None:
        super().__init__()
        self._model_type = model_type
        self._device = device
        self._batch_size = batch_size
        self._concept_penalty_coeff = concept_penalty_coeff
        self._meteor_coeff = meteor_coeff
        self._meteor_metric = MeteorMetric()

    def _get_missing_concepts(self, gen: str, concepts: List[str]):
        gen_text = gen.lower()
        missing_concepts = []
        for concept in concepts:
            if concept not in gen_text:
                missing_concepts.append(concept)
        return missing_concepts

    def _get_meteor_scores(self, gen: str, references: List[str]):
        scores = self._meteor_metric.compute(None, [gen], [references])
        return scores["lexical/meteor"][1]

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        done_n_missing_concepts = []
        done_meteor_scores = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)
                missing_concepts = self._get_missing_concepts(
                    gen, meta_info["concepts"]
                )
                done_n_missing_concepts.append(len(missing_concepts))
                done_meteor_scores.append(self._get_meteor_scores(gen, ref))

        # get pref scores
        pref_scores = get_scores(
            done_gen_texts, self._model_type, self._device, self._batch_size
        )

        # final
        rewards[done_ixs] = self._meteor_coeff * np.array(done_meteor_scores)
        rewards[done_ixs] += np.array(pref_scores) / (
            1 + self._concept_penalty_coeff * np.array(done_n_missing_concepts)
        )
        return rewards.tolist()


if __name__ == "__main__":
    import time

    rm = CommonGenPrefRM("11b", "cpu", 5)

    start = time.time()
    print(
        rm(
            [None, None, None],
            [
                "A man drives a car during a race at an event.",
                "A car drives a man during a race at an event.",
                "The event is a man driving a car race.",
            ],
            [None, None, None],
            [True, True, True],
            [None, None, None],
        )
    )
    end = time.time()
    print(end - start)

    start = time.time()
    print(
        rm(
            [None, None, None],
            [
                "A man drives a car during a race at an event.",
                "A car drives a man during a race at an event.",
                "The event is a man driving a car race.",
            ],
            [None, None, None],
            [True, True, True],
            [None, None, None],
        )
    )
    end = time.time()
    print(end - start)
