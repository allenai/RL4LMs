from typing import List, Dict, Any, Tuple
from transformers import PreTrainedModel
from accelerate import Accelerator
from rl4lms.envs.text_generation.metric import BaseMetric, MetricType
from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict
import numpy as np

class GenerationDataset(Dataset):
    def __init__(self, 
                 prompts: List[str], 
                 gen_texts: List[str],
                 ref_texts: List[str],
                 meta_infos: List[str]):
        super().__init__()
        self._prompts = prompts
        self._gen_texts = gen_texts
        self._ref_texts = ref_texts
        self._meta_infos = meta_infos
        self._size = len(self._prompts)

    def __len__(self):
        return self._size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
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
    for item in batch:
        prompts.append(item["prompt"])
        ref_texts.append(item["ref"])
        gen_texts.append(item["gen"])
        meta_infos.append(item["meta_info"])
    return prompts, gen_texts, ref_texts, meta_infos


def compute_single_metric(metric: BaseMetric,
                          prompts: List[str],
                          gen_texts: List[str],
                          ref_texts: List[str],
                          meta_infos: List[Dict[str, Any]],
                          model: PreTrainedModel,
                          split_name: str,
                          accelerator: Accelerator):
    # if it is not a distributed metric, run this only on the main process and return results
    if metric.metric_type == MetricType.NON_DIST:
        if accelerator.is_main_process:
            metric_results = metric.compute(prompts, gen_texts, ref_texts, meta_infos, model, split_name)
            return metric_results
        
    elif metric.metric_type == MetricType.DIST:
        dataset = GenerationDataset(prompts, gen_texts, ref_texts, meta_infos)
        batch_size = int(len(dataset) / accelerator.num_processes)
        dataloader = DataLoader(dataset=dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
        dataloader = accelerator.prepare(dataloader)

        all_corpus_level_scores = defaultdict(list)
        for batch_prompts, batch_gen_texts, batch_ref_texts, batch_meta_infos in dataloader:
            metric_dict = metric.compute(batch_prompts, batch_gen_texts, batch_ref_texts, batch_meta_infos, model, split_name)
            corpus_level_scores = {key: torch.tensor([value[1]]).to(accelerator.device) for key, value in metric_dict.items()}  # TBD : for now, lets work with only corpus scores
            gathered_corpus_level_scores = accelerator.gather_for_metrics(corpus_level_scores)
            for key, value in gathered_corpus_level_scores.items():
                all_corpus_level_scores[key].extend(value.tolist())

        
        # average the corpus level metrics
        # TBD: handle individual scores
        corpus_level_scores = {metric_name: (None, np.mean(scores).item()) for metric_name, scores in all_corpus_level_scores.items()}
        return corpus_level_scores

    # else create a dataloader out of all the samples
    return {}

