from rl4lms.data_pools.text_generation_pool import TextGenPool, Sample
from rl4lms.data_pools.task_utils.totto import preprocess_utils
from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import os
from urllib.request import urlretrieve
from pathlib import Path
import pandas
from collections import defaultdict
import zipfile
import json


class ToTTo(TextGenPool):
    @classmethod
    def prepare(cls, split: str,
                representation: str = 'subtable',
                **args) -> 'TextGenPool':
        ds = load_dataset('totto')
        samples = []
        split_id = ToTTo.gen_split_name(split)
        n_samples = len(ds[split_id])
        for ix, item in tqdm(enumerate(ds[split_id]),
                             desc="Loading ToTTo dataset", total=n_samples):

            table = item["table"]
            table_page_title = item["table_page_title"]
            table_section_title = item["table_section_title"]
            cell_indices = item["highlighted_cells"]

            # can potentially add additional targets for various stages of annotation here
            targets = item["sentence_annotations"]["final_sentence"]

            if split_id == "test":
                targets = [""]  # empty references instead of none

            subtable = (
                preprocess_utils.get_highlighted_subtable(
                    table=table,
                    cell_indices=cell_indices,
                    with_heuristic_headers=True))

            if representation == 'subtable':
                subtable_str = (
                    preprocess_utils.linearize_subtable(
                        subtable=subtable,
                        table_page_title=None,
                        table_section_title=None))

                subtable_metadata_str = (
                    preprocess_utils.linearize_subtable(
                        subtable=subtable,
                        table_page_title=table_page_title,
                        table_section_title=table_section_title))

                prompt = subtable_str + subtable_metadata_str
            elif representation == 'fulltable':
                full_table_str = preprocess_utils.linearize_full_table(
                    table=table,
                    cell_indices=cell_indices,
                    table_page_title=None,
                    table_section_title=None)

                full_table_metadata_str = (
                    preprocess_utils.linearize_full_table(
                        table=table,
                        cell_indices=cell_indices,
                        table_page_title=table_page_title,
                        table_section_title=table_section_title))

                prompt = full_table_str + full_table_metadata_str
            else:
                raise NotImplementedError

            # reformat sentence annotations (to fit in official totto eval)
            reformatted_sent_annotations = []
            n_refs = len(targets)
            for ref_ix in range(n_refs):
                annotation = {
                    "original_sentence": item["sentence_annotations"]["original_sentence"][ref_ix],
                    "sentence_after_deletion": item["sentence_annotations"]["sentence_after_deletion"][ref_ix],
                    "sentence_after_ambiguity": item["sentence_annotations"]["sentence_after_ambiguity"][ref_ix],
                    "final_sentence": item["sentence_annotations"]["final_sentence"][ref_ix],
                }
                reformatted_sent_annotations.append(annotation)
            item["sentence_annotations"] = reformatted_sent_annotations

            # change overlap_subset to bool type
            item["overlap_subset"] = True if item["overlap_subset"] == "True" else False

            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=targets,
                            meta_data={
                                "raw_table": item
                            }
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "validation"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name


class CommonGen(TextGenPool):
    @classmethod
    def prepare(cls, split: str,
                concept_separator_token: str = " ",
                concept_end_token=" ",
                prefix: str = "summarize: ") -> 'TextGenPool':
        ds = load_dataset("gem", "common_gen")
        samples = []
        split_id = CommonGen.gen_split_name(split)
        for ix, item in enumerate(ds[split_id]):
            concepts = concept_separator_token.join(item["concepts"])
            concepts = prefix + concepts
            concepts += concept_end_token
            if item["target"] == "":
                # just to avoid breaking of metric computation
                item["target"] = "empty reference"
            targets = [item["target"]]
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=concepts,
                            references=targets,
                            meta_data={
                                "concepts": item["concepts"]
                            }
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "validation"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name


class Xsum(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt_suffix: str = "TL;DR:"):
        dataset = load_dataset("gem", "xsum")
        dataset_split = dataset[split]
        samples = []
        for ix, item in enumerate(dataset_split):
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=item["document"] +
                            prompt_suffix,
                            references=[item["target"]]
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance


class CNNDailyMail(TextGenPool):
    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                truncate_article: int = None,
                max_size: int = None):
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        dataset_split = CommonGen.gen_split_name(split)
        samples = []
        for ix, item in tqdm(enumerate(dataset[dataset_split]),
                             desc="Tokenizing dataset",
                             total=len(dataset[dataset_split])):

            if truncate_article is not None:
                tokens = word_tokenize(item["article"])
                tokens = tokens[:truncate_article]
                item["article"] = " ".join(tokens)

            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt_prefix +
                            item["article"] + prompt_suffix,
                            references=[item["highlights"]]
                            )
            samples.append(sample)

            if max_size is not None and ix == (max_size-1):
                break

        pool_instance = cls(samples)
        return pool_instance


class IMDB(TextGenPool):
    """
    IMDB Dataset for sentiment continuation task
    """
    @classmethod
    def prepare(cls, split: str):
        dataset = load_dataset("imdb")
        if split in ["train", "val"]:
            dataset_split = dataset["train"].shuffle()
            train_ratio = 0.8
            train_index = int(len(dataset_split) * train_ratio)
            dataset_split = dataset_split[:train_index] if split == "train" else dataset_split[train_index:]
        else:
            dataset_split = dataset[split].shuffle()
            dataset_split = dataset_split[:5000]

        samples = []
        for ix, text in enumerate(dataset_split["text"]):

            # here we consider 50% of tokens as prompt
            prompt_text = text.split(" ")
            prompt_text = " ".join(prompt_text[:int(len(prompt_text) * 0.5)])

            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt_text,
                            references=[text]
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance


class IMDBForSeq2Seq(TextGenPool):
    """
    IMDB Dataset in seq2seq format to train supervised generator
    """
    @classmethod
    def prepare(cls, split: str, positive_ratio: int = 1.0):
        dataset = load_dataset("imdb")
        if split in ["train", "val"]:
            dataset_split = dataset["train"].shuffle()
            train_ratio = 0.8
            train_index = int(len(dataset_split) * train_ratio)
            dataset_split = dataset_split[:train_index] if split == "train" else dataset_split[train_index:]
        else:
            # limit test to 5000
            dataset_split = dataset[split].shuffle()
            dataset_split = dataset_split[:5000]

        samples = []
        for ix, (text, label) in enumerate(zip(dataset_split["text"], dataset_split["label"])):

            # here we consider 50% of tokens as prompt and rest as references
            tokenized_text = text.split(" ")
            text_split_index = int(len(tokenized_text) * 0.5)
            prompt_text = " ".join(tokenized_text[:text_split_index])
            ref_text = " ".join(tokenized_text[text_split_index:])

            # add only positive examples for train set
            if split == "train" and label == 1 or split != "train":
                sample = Sample(id=f"{split}_{ix}",
                                prompt_or_input_text=prompt_text,
                                references=[ref_text],
                                meta_data={
                                    "reference": text
                                }
                                )
                samples.append(sample)

        # truncate train split
        if split == "train":
            samples = samples[:int(len(samples) * positive_ratio)]

        pool_instance = cls(samples)
        return pool_instance


def download_file_using_url(url: str, dest_path: str):
    urlretrieve(url, dest_path)


class NarrativeQA(TextGenPool):
    @classmethod
    def normalize_text(cls, text, strip: bool):
        # https: // github.com/allenai/unifiedqa/blob/7bf0653c6fb68a51019924fd4c51615155acbebe/tasks.py  # L54-L58
        text = text.lower()
        if strip:
            text = text.strip()
        text.replace("'", '')
        return text

    @classmethod
    def prepare(cls, split: str):

        # URLs
        urls = {
            "train": "https://storage.googleapis.com/unifiedqa/data/narrativeqa/train.tsv",
            "val": "https://storage.googleapis.com/unifiedqa/data/narrativeqa/dev.tsv",
            "test": "https://storage.googleapis.com/unifiedqa/data/narrativeqa/test.tsv"
        }

        # destination path
        dest_base_path = os.path.join(Path.home(), "narrative_qa")
        os.makedirs(dest_base_path, exist_ok=True)

        # download the file
        split_path = os.path.join(dest_base_path, f"{split}.tsv")
        if not os.path.exists(split_path):
            download_file_using_url(urls[split], split_path)

        # load the split
        split_df = pandas.read_csv(split_path, sep='\t',
                                   header=None, encoding="utf-8")

        # group questions and answers
        prompts_and_answers = defaultdict(list)
        for _, (prompt, answer) in split_df.iterrows():
            prompts_and_answers[prompt].append(answer)

        samples = []
        for ix, (prompt, answers) in enumerate(prompts_and_answers.items()):
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=cls.normalize_text(
                                prompt, False),
                            references=[cls.normalize_text(answer, True) for answer in answers])
            samples.append(sample)

        dp_instance = cls(samples)
        return dp_instance


class WMT(TextGenPool):
    @classmethod
    def get_dataset(cls, wmt_id: str, source_language: str, target_language: str, split: str):
        try:
            language_pair = f"{source_language}-{target_language}"
            dataset = load_dataset(f"{wmt_id}", language_pair)
        except:
            language_pair = f"{target_language}-{source_language}"
            dataset = load_dataset(f"{wmt_id}", language_pair)
        dataset = dataset[split]
        return dataset

    @classmethod
    def prepare(cls,
                wmt_id: str,
                split: str,
                source_language: str,
                target_language: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = WMT.get_dataset(
            wmt_id, source_language, target_language, dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"][source_language] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"][target_language]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


class WMT14PreprocessedEnDe(TextGenPool):
    @classmethod
    def get_dataset(cls, split: str):
        dataset = load_dataset("stas/wmt14-en-de-pre-processed")[split]
        return dataset

    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = WMT14PreprocessedEnDe.get_dataset(dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"]["en"] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"]["de"]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


class WMT16NewsOnlyDatasetEnDe(TextGenPool):
    @classmethod
    def get_dataset(cls, split: str):

        if split == "train":
            dataset = load_dataset("news_commentary", "de-en")[split]
            return dataset
        else:
            dataset = load_dataset("wmt16", "de-en")[split]
        return dataset

    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = WMT16NewsOnlyDatasetEnDe.get_dataset(dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"]["en"] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"]["de"]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


class IWSLT2017EnDe(TextGenPool):
    @classmethod
    def get_dataset(cls, split: str):
        dataset = load_dataset(
            "iwslt2017", "iwslt2017-de-en", ignore_verifications=True)
        return dataset[split]

    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                ):
        dataset_split = CommonGen.gen_split_name(split)
        dataset = IWSLT2017EnDe.get_dataset(dataset_split)
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                             desc="Preparing dataset",
                             total=len(dataset)):

            prompt = prompt_prefix + \
                item["translation"]["en"] + prompt_suffix
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt,
                            references=[item["translation"]["de"]]
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


class CRD3DialogueGeneration(TextGenPool):
    SOURCE_URL = "https://github.com/RevanthRameshkumar/CRD3/archive/refs/heads/master.zip"
    DEST_BASE_FOLDER = "crd3"
    DEST_EXTRACTED_FOLDER = "CRD3-master"
    ZIP_FILE_NAME = "master.zip"
    PATH_TO_ALIGNED_DATA = "data/aligned data"
    PATH_TO_CLEANED_DATA = "data/cleaned data"

    @classmethod
    def prepare(cls, split: str, max_context_size: int):
        dest_base_path = os.path.join(
            Path.home(), CRD3DialogueGeneration.DEST_BASE_FOLDER)
        path_to_extracted_folder = os.path.join(
            dest_base_path, CRD3DialogueGeneration.DEST_EXTRACTED_FOLDER)
        path_to_zip_file = os.path.join(
            dest_base_path, CRD3DialogueGeneration.ZIP_FILE_NAME)

        # download and extract if it does not exist
        if not os.path.exists(path_to_extracted_folder):
            os.makedirs(dest_base_path, exist_ok=True)
            download_file_using_url(
                CRD3DialogueGeneration.SOURCE_URL, path_to_zip_file)

            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(dest_base_path)

        # get all train, val and test episode files
        ep_file_paths = {
            "train": os.path.join(path_to_extracted_folder, CRD3DialogueGeneration.PATH_TO_ALIGNED_DATA, "train_files"),
            "val": os.path.join(path_to_extracted_folder, CRD3DialogueGeneration.PATH_TO_ALIGNED_DATA, "val_files"),
            "test": os.path.join(path_to_extracted_folder, CRD3DialogueGeneration.PATH_TO_ALIGNED_DATA, "test_files"),
        }

        with open(ep_file_paths[split], "r") as fp:
            ep_file_names = fp.read().splitlines()

        # now, load all episode files for the split
        samples = []
        for ep_ix, file_name in enumerate(ep_file_names):
            ep_path = os.path.join(
                path_to_extracted_folder, CRD3DialogueGeneration.PATH_TO_CLEANED_DATA, f"{file_name}.json")
            with open(ep_path, "r") as fp:
                ep_data = json.load(fp)

            # for each utterance, keep track of all the utterances before to get the context
            # initially empty for each episode
            contexts = []
            for turn_ix, turn_data in enumerate(ep_data["TURNS"]):
                # current utterance string
                names = turn_data["NAMES"][0]
                turn_utterances = " ".join(turn_data["UTTERANCES"])

                # sample
                if len(contexts) > 1:
                    contexts_shortened = contexts[-max_context_size:]
                    prompt_or_input = "\n".join(contexts_shortened)

                    # add name to the prompt
                    prompt_or_input = prompt_or_input + "\n" + names + ": "

                    sample = Sample(id=f"{split}_ep_{ep_ix}_turn_{turn_ix}",
                                    prompt_or_input_text=prompt_or_input,
                                    references=[turn_utterances])
                    samples.append(sample)

                # update the context
                turn_string_full = names + ": " + turn_utterances
                contexts.append(turn_string_full)

        dp_instance = cls(samples)
        return dp_instance


class DailyDialog(TextGenPool):
    EOU_TOKEN = "<EOU>"
    @classmethod
    def prepare(cls, split: str, context_size: int):
        split = CommonGen.gen_split_name(split)
        dataset = load_dataset("daily_dialog", split=split)
        samples = []
        utterance_id = 0
        for item in dataset:
            contexts = []
            for utterance, emotion, intent in zip(item["dialog"],
                                                  item["emotion"],
                                                  item["act"]):
                if len(contexts) >= context_size:
                    context = DailyDialog.EOU_TOKEN.join(contexts[-context_size:]) 
                    context += " " + DailyDialog.EOU_TOKEN
                    target = utterance + DailyDialog.EOU_TOKEN
                    sample = Sample(id=utterance_id, 
                                    prompt_or_input_text=context, 
                                    references=[target],
                                    meta_data={
                                        "emotion": [emotion],
                                        "intent": [intent]
                                    })
                    samples.append(sample)
                contexts.append(utterance)
                utterance_id += 1

        dp_instance = cls(samples)
        return dp_instance


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import numpy as np
    dp = DailyDialog.prepare("val", 5)
    print(dp[0])
    