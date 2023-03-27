from typing import List, Dict, Any
from transformers import PreTrainedModel
from accelerate import Accelerator
from rl4lms.envs.text_generation.metric import BaseMetric, MetricType
from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict
import numpy as np

# sub token for sample metric score if it is none
SAMPLE_METRIC_SCORE_SUB_TOKEN = int(1e+9)


class GenerationDataset(Dataset):
    def __init__(self, 
                 sample_ids: List[str],
                 prompts: List[str],
                 gen_texts: List[str],
                 ref_texts: List[str],
                 meta_infos: List[str]):
        super().__init__()
        self._sample_ids = sample_ids
        self._prompts = prompts
        self._gen_texts = gen_texts
        self._ref_texts = ref_texts
        self._meta_infos = meta_infos
        self._size = len(self._prompts)

    def __len__(self):
        return self._size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "sample_id": self._sample_ids[idx],
            "prompt": self._prompts[idx],
            "gen": self._gen_texts[idx],
            "ref": self._ref_texts[idx],
            "meta_info": self._meta_infos[idx]
        }
        return item

def collate_fn(batch: List[Dict[str, Any]]):
    prompts = []
    ref_texts = []
    gen_texts = []
    meta_infos = []
    sample_ids = []
    for item in batch:
        prompts.append(item["prompt"])
        ref_texts.append(item["ref"])
        gen_texts.append(item["gen"])
        meta_infos.append(item["meta_info"])
        sample_ids.append(item["sample_id"])
    return sample_ids, prompts, gen_texts, ref_texts, meta_infos


def prepare_sample_scores(metric_dict: Dict[str, Any], size: int, device: str):
    """
    It prepares sample level metric scores for gathering across processes
    """
    sample_scores_by_metric = {}
    for metric_name, (individual_scores, _) in metric_dict.items():
        if individual_scores is None:
            sample_scores = torch.tensor([SAMPLE_METRIC_SCORE_SUB_TOKEN] * size).to(device)
        else:
            sample_scores = torch.tensor(individual_scores).to(device)
        sample_scores_by_metric[metric_name] = sample_scores
    return sample_scores_by_metric

def compute_single_metric(metric: BaseMetric,
                          sample_ids: List[str],
                          prompts: List[str],
                          gen_texts: List[str],
                          ref_texts: List[str],
                          meta_infos: List[Dict[str, Any]],
                          model: PreTrainedModel,
                          split_name: str,
                          accelerator: Accelerator):
    # if it is not a distributed metric, run this only on the main process and return results
    if metric.metric_type == MetricType.NON_DIST and accelerator.is_main_process:
        metric_results = metric.compute(prompts, gen_texts, ref_texts, meta_infos, model, split_name)
        return metric_results
        
    elif metric.metric_type == MetricType.DIST:
        dataset = GenerationDataset(sample_ids, prompts, gen_texts, ref_texts, meta_infos)
        batch_size = int(len(dataset) / accelerator.num_processes)
        dataloader = DataLoader(dataset=dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
        dataloader = accelerator.prepare(dataloader)

        all_corpus_level_scores = defaultdict(list)
        sample_level_scores_by_sample_id = {sample_id: {} for sample_id in sample_ids}
        for batch_sample_ids, batch_prompts, batch_gen_texts, batch_ref_texts, batch_meta_infos in dataloader:
            metric_dict = metric.compute(batch_prompts, batch_gen_texts, batch_ref_texts, batch_meta_infos, model, split_name)
            
            # gather corpus level scores
            corpus_level_scores = {key: torch.tensor([value[1]]).to(accelerator.device) for key, value in metric_dict.items()}
            gathered_corpus_level_scores = accelerator.gather_for_metrics(corpus_level_scores)
            for key, value in gathered_corpus_level_scores.items():
                all_corpus_level_scores[key].extend(value.tolist())
            
            # gather sample level scores
            batch_sample_ids = torch.tensor(batch_sample_ids).to(accelerator.device)
            gathered_sample_ids = accelerator.gather_for_metrics(batch_sample_ids).tolist()
            batch_sample_scores = prepare_sample_scores(metric_dict, len(batch_sample_ids), accelerator.device)
            gathered_sample_scores = accelerator.gather_for_metrics(batch_sample_scores)
            for metric_name, sample_scores in gathered_sample_scores.items():
                for sample_id, sample_score in zip(gathered_sample_ids, sample_scores.tolist()):
                    sample_level_scores_by_sample_id[sample_id][metric_name] = sample_score

        # consolidate both sample and corpus level scores
        final_metrics = {}
        for metric_name in metric_dict.keys():
            sample_level_scores = ["n/a" if sample_level_scores_by_sample_id[sample_id][metric_name] == SAMPLE_METRIC_SCORE_SUB_TOKEN 
                                   else sample_level_scores_by_sample_id[sample_id][metric_name] 
                                   for sample_id in sample_ids]

            # if sample level scores, if one of them is sub token which was added during gathering
            if SAMPLE_METRIC_SCORE_SUB_TOKEN in sample_level_scores:
                sample_level_scores = []

            corpus_score = np.mean(all_corpus_level_scores[metric_name]).item()
            final_metrics[metric_name] = (sample_level_scores, corpus_score)

        return final_metrics
    else:
        return {}

