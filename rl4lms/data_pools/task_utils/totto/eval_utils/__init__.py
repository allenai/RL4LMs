import subprocess
import os
from tempfile import TemporaryDirectory
import jsonlines
from typing import List
import json


def compute_parent(predicted_texts: List[str],
                   raw_tables: List[dict]):

    with TemporaryDirectory() as temp_dir:

        # write tables
        target_path = os.path.join(temp_dir, "samples.jsonl")
        with jsonlines.open(target_path, "w") as writer:
            for table in raw_tables:
                writer.write(table)

        # write gen texts
        prediction_path = os.path.join(temp_dir, "predictions.txt")
        with open(prediction_path, "w") as fp:
            predicted_texts = '\n'.join(predicted_texts)
            fp.write(predicted_texts)

        cmd = ['bash', 'totto_parent_eval.sh',
               '-p', prediction_path,
               '-t', target_path,
               '--output_dir', temp_dir,
               ]
        subprocess.check_call(cmd,
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              stdout=subprocess.DEVNULL)

        # read the results back
        with open(os.path.join(temp_dir, "parent_overall.json")) as fp:
            parent_overall_results = json.load(fp)

        with open(os.path.join(temp_dir, "parent_overlap.json")) as fp:
            parent_overlap_results = json.load(fp)

        with open(os.path.join(temp_dir, "parent_non_overlap.json")) as fp:
            parent_non_overlap_results = json.load(fp)

        return parent_overall_results, parent_overlap_results, parent_non_overlap_results


def compute_bleu(predicted_texts: List[str],
                 raw_tables: List[dict]):

    def _read_results(path):
        try:
            with open(path) as fp:
                score = json.load(fp)["score"]/100
        except:
            score = 0.0
        return score

    with TemporaryDirectory() as temp_dir:

        # write tables
        target_path = os.path.join(temp_dir, "samples.jsonl")
        with jsonlines.open(target_path, "w") as writer:
            for table in raw_tables:
                writer.write(table)

        # write gen texts
        prediction_path = os.path.join(temp_dir, "predictions.txt")
        with open(prediction_path, "w") as fp:
            predicted_texts = '\n'.join(predicted_texts)
            fp.write(predicted_texts)

        cmd = ['bash', 'totto_bleu_eval.sh',
               '-p', prediction_path,
               '-t', target_path,
               '--output_dir', temp_dir,
               ]
        subprocess.check_call(cmd,
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              stdout=subprocess.DEVNULL)

        # read the results back
        bleu_overall = _read_results(
            os.path.join(temp_dir, "bleu_overall.json"))
        bleu_overlap = _read_results(
            os.path.join(temp_dir, "bleu_overlap.json"))
        bleu_non_overlap = _read_results(
            os.path.join(temp_dir, "bleu_non_overlap.json"))
        return bleu_overall, bleu_overlap, bleu_non_overlap
