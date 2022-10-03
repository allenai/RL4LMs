# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Script to compute PARENT metric modified for ToTTo dataset.

https://arxiv.org/abs/1906.01081

Modified from:
https://github.com/google-research/language/tree/master/language/table_text_eval


The <reference_file> and <generation_file> should contain references and
generations, respectively, one per line. The <table_file> should contain the
ground truth tables corresponding to these in each line.

For multiple references, each reference should be in a separate file and named
like: '<reference-path>0', '<reference-path>1',.....

The format for tables is pairs of attributes and values:
  attribute_1|||value_1<TAB>attribute_2|||value_2<TAB>...

The entailment probability is computing using word overlap.
"""
import collections
import glob
import io
import math
import os
import json
import numpy as np

from absl import app
from absl import flags
import sacrebleu

try:
    import sacrebleu.tokenizers.tokenizer_13a  # pylint: disable=g-import-not-at-top
except ImportError:
    raise ValueError("Latest version of sacrebleu is missing.")

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "reference_path", None, "Text file containing references, one per line. "
    "Multiple references should be in different files named as: "
    "'<reference_path>0', '<reference_path>1', '<reference_path>2'.")

flags.DEFINE_string("generation_path", None,
                    "Text file containing generations, one per line.")

flags.DEFINE_string(
    "precision_table_path", None,
    "Text file containing tables for computing precision, one per line.")

flags.DEFINE_string(
    "recall_table_path", None,
    "Text file containing tables for computing recall, one per line.")

flags.DEFINE_string(
    "result_path", None,
    "Outputs will be written to this file")

flags.DEFINE_float("smoothing", 0.00001,
                   "Constant to replace 0 precision and recall scores with.")

flags.DEFINE_float("lambda_weight", None,
                   "Weighting factor for recall computed against the table.")


def _normalize_text(s):
    # pylint: disable=unnecessary-lambda
    def tokenize_fn(
        x): return sacrebleu.tokenizers.tokenizer_13a.Tokenizer13a()(x)
    return tokenize_fn(s.strip().lower())


def _text_reader(text_file):
    """Returns list of lines from the text file.

    Performs lowercasing and white-space tokenization on each line before
    returning.

    Args:
      text_file: String filename.
    """
    texts = []
    with io.open(text_file, encoding="utf-8") as f:
        for line in f:
            line = _normalize_text(line)
            texts.append(line.split())
    return texts


def _text_reference_reader(text_file):
    """Returns list of references in both the single and multi-reference setting.

    Performs lowercasing and white-space tokenization on each line before
    returning.

    Args:
      text_file: String filename.
    """
    single_reference_exists = os.path.isfile(text_file)

    # Check for multi-references.
    multi_reference_paths = glob.glob(text_file + "[0-9]")

    # Either the file should exist or it should correspond to multiple reference
    # files but not both.
    assert ((single_reference_exists or multi_reference_paths) and
            (not single_reference_exists or not multi_reference_paths))

    if single_reference_exists:
        references = _text_reader(text_file)
        references = [[x] for x in references]
        return references
    else:
        # In the multi-reference case, we should have 3 reference files.
        assert len(multi_reference_paths) == 3
        references0 = _text_reader(multi_reference_paths[0])
        references1 = _text_reader(multi_reference_paths[1])
        references2 = _text_reader(multi_reference_paths[2])

        assert len(references0) == len(references1)
        assert len(references0) == len(references2)
        multi_references = []
        for i in range(len(references0)):
            multi_reference = [references0[i], references1[i], references2[i]]
            multi_references.append(multi_reference)

        return multi_references


def _table_reader(table_file):
    """Yields tables from the table file.

    Tables are parsed into a list of tuples with tokenized entries.

    Args:
      table_file: String filename.
    """
    with io.open(table_file, encoding="utf-8") as f:
        for line in f:
            entries = line.lower().split("\t")
            # pylint: disable=g-complex-comprehension
            table = [[
                _normalize_text(member).split() for member in entry.split("|||")
            ] for entry in entries]
            yield table


def overlap_probability(ngram, table, smoothing=0.0, stopwords=None):
    """Returns the probability that the given n-gram overlaps with the table.

    A simple implementation which checks how many tokens in the n-gram are also
    among the values in the table. For tables with (attribute, value) pairs on the
    `value` field is condidered. For tables with (head, relation, tail) triples a
    concatenation of `head` and `tail` are considered.

    E.g.:
      >>> overlap_probability(["michael", "dahlquist"],
                               [(["name"], ["michael", "dahlquist"])])
      >>> 1.0

    Args:
      ngram: List of tokens.
      table: List of either (attribute, value) pairs or (head, relation, tail)
        triples. Each member of the pair / triple is assumed to already be
        tokenized into a list of strings.
      smoothing: (Optional) Float parameter for laplace smoothing.
      stopwords: (Optional) List of stopwords to ignore (assign P = 1).

    Returns:
      prob: Float probability of ngram being entailed by the table.
    """
    # pylint: disable=g-complex-comprehension
    if len(table[0]) == 2:
        table_values = set([tok for _, value in table for tok in value])
    else:
        table_values = set(
            [tok for head, _, tail in table for tok in head + tail])
    overlap = 0
    for token in ngram:
        if stopwords is not None and token in stopwords:
            overlap += 1
            continue
        if token in table_values:
            overlap += 1
    return float(overlap + smoothing) / float(len(ngram) + smoothing)


def _mention_probability(table_entry, sentence, smoothing=0.0):
    """Returns the probability that the table entry is mentioned in the sentence.

    A simple implementation which checks the longest common subsequence between
    the table entry and the sentence. For tables with (attribute, value) pairs
    only the `value` is considered. For tables with (head, relation, tail) triples
    a concatenation of the `head` and `tail` is considered.

    E.g.:
      >>> _mention_probability((["name"], ["michael", "dahlquist"]),
                               ["michael", "dahlquist", "was", "a", "drummer"])
      >>> 1.0

    Args:
      table_entry: Tuple of either (attribute, value) or (head, relation, tail).
        Each member of the tuple is assumed to already be tokenized into a list of
        strings.
      sentence: List of tokens.
      smoothing: Float parameter for laplace smoothing.

    Returns:
      prob: Float probability of entry being in sentence.
    """
    if len(table_entry) == 2:
        value = table_entry[1]
    else:
        value = table_entry[0] + table_entry[2]
    overlap = _len_lcs(value, sentence)
    return float(overlap + smoothing) / float(len(value) + smoothing)


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.

    The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _ngrams(sequence, order):
    """Yields all ngrams of given order in sequence."""
    assert order >= 1
    for n in range(order, len(sequence) + 1):
        yield tuple(sequence[n - order: n])


def _ngram_counts(sequence, order):
    """Returns count of all ngrams of given order in sequence."""
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(_ngrams(sequence, order))


def parent(predictions,
           references,
           precision_tables,
           recall_tables,
           lambda_weight=0.5,
           smoothing=0.00001,
           max_order=4,
           entailment_fn=overlap_probability,
           mention_fn=_mention_probability):
    """Metric for comparing predictions to references given tables.

    Args:
      predictions: An iterator over tokenized predictions. Each prediction is a
        list.
      references: An iterator over lists of tokenized references. Each prediction
        can have multiple references.
      precision_tables: An iterator over the tables. Each table is a list of
        tuples, where a tuple can either be (attribute, value) pair or (head,
        relation, tail) triple. The members of the tuples are assumed to be
        themselves tokenized lists of strings. E.g. `[(["name"], ["michael",
        "dahlquist"]), (["birth", "date"], ["december", "22", "1965"])]` is one
        table in the (attribute, value) format with two entries.
      recall_tables: An iterator over the tables. Each table is a list of tuples,
        where a tuple can either be (attribute, value) pair or (head, relation,
        tail) triple. The members of the tuples are assumed to be themselves
        tokenized lists of strings. E.g. `[(["name"], ["michael", "dahlquist"]),
        (["birth", "date"], ["december", "22", "1965"])]` is one table in the
        (attribute, value) format with two entries.
      lambda_weight: Float weight in [0, 1] to multiply table recall.
      smoothing: Float value for replace zero values of precision and recall.
      max_order: Maximum order of the ngrams to use.
      entailment_fn: A python function for computing the probability that an
        ngram is entailed by the table. Its signature should match that of
        `overlap_probability` above.
      mention_fn: A python function for computing the probability that a
        table entry is mentioned in the text. Its signature should
          match that of `_mention_probability` above.

    Returns:
      precision: Average precision of all predictions.
      recall: Average recall of all predictions.
      f1: Average F-scores of all predictions.
      all_f_scores: List of all F-scores for each item.
    """
    precisions, recalls, all_f_scores = [], [], []
    reference_recalls, table_recalls = [], []
    all_lambdas = []
    for prediction, list_of_references, precision_table, recall_table in zip(
            predictions, references, precision_tables, recall_tables):
        c_prec, c_rec, c_f = [], [], []
        ref_rec, table_rec = [], []

        # Assert that either single reference or 3 references.
        assert len(list_of_references) == 1 or len(list_of_references) == 3

        for reference in list_of_references:
            # Weighted ngram precisions and recalls for each order.
            ngram_prec, ngram_rec = [], []
            for order in range(1, max_order + 1):
                # Collect n-grams and their entailment probabilities.
                pred_ngram_counts = _ngram_counts(prediction, order)
                pred_ngram_weights = {
                    ngram: entailment_fn(ngram, precision_table)
                    for ngram in pred_ngram_counts
                }
                ref_ngram_counts = _ngram_counts(reference, order)
                ref_ngram_weights = {
                    ngram: entailment_fn(ngram, precision_table)
                    for ngram in ref_ngram_counts
                }

                # Precision.
                numerator, denominator = 0., 0.
                for ngram, count in pred_ngram_counts.items():
                    denominator += count
                    prob_ngram_in_ref = min(
                        1., float(ref_ngram_counts.get(ngram, 0) / count))
                    numerator += count * (
                        prob_ngram_in_ref +
                        (1. - prob_ngram_in_ref) * pred_ngram_weights[ngram])
                if denominator == 0.:
                    # Set precision to 0.
                    ngram_prec.append(0.0)
                else:
                    ngram_prec.append(numerator / denominator)

                # Recall.
                numerator, denominator = 0., 0.
                for ngram, count in ref_ngram_counts.items():
                    prob_ngram_in_pred = min(
                        1., float(pred_ngram_counts.get(ngram, 0) / count))
                    denominator += count * ref_ngram_weights[ngram]
                    numerator += count * \
                        ref_ngram_weights[ngram] * prob_ngram_in_pred
                if denominator == 0.:
                    # Set recall to 1.
                    ngram_rec.append(1.0)
                else:
                    ngram_rec.append(numerator / denominator)

            # Compute recall against table fields.
            table_mention_probs = [
                mention_fn(entry, prediction) for entry in recall_table
            ]
            table_rec.append(sum(table_mention_probs) / len(recall_table))

            # Smoothing.
            for order in range(1, max_order):
                if ngram_prec[order] == 0.:
                    ngram_prec[order] = smoothing
                if ngram_rec[order] == 0.:
                    ngram_rec[order] = smoothing

            # Compute geometric averages of precision and recall for all orders.
            w = 1. / max_order
            if any(prec == 0. for prec in ngram_prec):
                c_prec.append(0.)
            else:
                sp = (w * math.log(p_i) for p_i in ngram_prec)
                c_prec.append(math.exp(math.fsum(sp)))
            if any(rec == 0. for rec in ngram_rec):
                ref_rec.append(smoothing)
            else:
                sr = [w * math.log(r_i) for r_i in ngram_rec]
                ref_rec.append(math.exp(math.fsum(sr)))

            # Combine reference and table recalls.
            if table_rec[-1] == 0.:
                table_rec[-1] = smoothing
            if ref_rec[-1] == 0. or table_rec[-1] == 0.:
                c_rec.append(0.)
            else:
                if lambda_weight is None:
                    lw = sum([mention_fn(entry, reference) for entry in recall_table
                              ]) / len(recall_table)
                    lw = 1. - lw
                else:
                    lw = lambda_weight
                all_lambdas.append(lw)
                c_rec.append(
                    math.exp((1. - lw) * math.log(ref_rec[-1]) +
                             (lw) * math.log(table_rec[-1])))

            # F-score.
            c_f.append((2. * c_prec[-1] * c_rec[-1]) /
                       (c_prec[-1] + c_rec[-1] + 1e-8))

        # Get index of best F-score.
        max_i = max(enumerate(c_f), key=lambda x: x[1])[0]
        precisions.append(c_prec[max_i])
        recalls.append(c_rec[max_i])
        all_f_scores.append(c_f[max_i])
        reference_recalls.append(ref_rec[max_i])
        table_recalls.append(table_rec[max_i])

    avg_precision = sum(precisions) / \
        len(precisions) if len(precisions) > 0 else 0.0
    avg_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0.0
    avg_f_score = sum(all_f_scores) / \
        len(all_f_scores) if len(all_f_scores) > 0 else 0.0

    return avg_precision, avg_recall, avg_f_score, all_f_scores


def main(_):
    references = _text_reference_reader(FLAGS.reference_path)
    generations = _text_reader(FLAGS.generation_path)
    precision_tables = list(_table_reader(FLAGS.precision_table_path))
    recall_tables = list(_table_reader(FLAGS.recall_table_path))

    assert len(references) == len(generations)
    assert len(references) == len(precision_tables)
    assert len(references) == len(recall_tables)

    precision, recall, f_score, all_f = parent(
        generations,
        references,
        precision_tables,
        recall_tables,
        lambda_weight=FLAGS.lambda_weight,
        smoothing=FLAGS.smoothing,
        entailment_fn=overlap_probability)

    print("Evaluated %d examples." % len(all_f))
    print("Precision = %.2f Recall = %.2f F-score = %.2f" %
          (precision, recall, f_score))

    with open(FLAGS.result_path, "w") as fp:
        json.dump({
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "all_f": all_f if len(all_f) > 0 else None
        }, fp)


if __name__ == "__main__":
    flags.mark_flags_as_required([
        "reference_path", "generation_path", "precision_table_path",
        "recall_table_path", "result_path"
    ])
    app.run(main)
