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
"""Processes references for eval (except for tokenization)."""
import json
import os

from absl import app
from absl import flags

from rl4lms.data_pools.task_utils.totto.eval_utils import table_to_text_utils
import six

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input json file.")
flags.DEFINE_string("output_dir", None, "Output directory.")
flags.DEFINE_string("mode", None, "Either 'dev', or 'test'")


def get_references(json_example, mode="dev"):
    """Get references from json example."""
    multi_reference = []
    for annotation in json_example["sentence_annotations"]:
        final_sentence = annotation["final_sentence"]
        multi_reference.append(final_sentence)

    if mode == "dev" or mode == "test":
        while len(multi_reference) < 3:
            multi_reference.append("<null>")

    if mode == "dev" or mode == "test":
        if json_example["overlap_subset"]:
            multi_overlap_reference = multi_reference
            multi_nonoverlap_reference = None
        else:
            multi_nonoverlap_reference = multi_reference
            multi_overlap_reference = None
    else:
        multi_overlap_reference = None
        multi_nonoverlap_reference = None

    return multi_reference, multi_overlap_reference, multi_nonoverlap_reference


def get_parent_tables(json_example, mode="dev"):
    """Get tables in PARENT format for each json example."""
    table = json_example["table"]
    table_page_title = json_example["table_page_title"]
    table_section_title = json_example["table_section_title"]
    table_section_text = json_example["table_section_text"]

    cell_indices = json_example["highlighted_cells"]
    highlighted_subtable = (
        table_to_text_utils.get_highlighted_subtable(
            table=table, cell_indices=cell_indices))

    # Get PARENT format code.
    table_prec = table_to_text_utils.get_table_parent_format(
        table=table,
        table_page_title=table_page_title,
        table_section_title=table_section_title,
        table_section_text=table_section_text)

    table_rec = table_to_text_utils.get_subtable_parent_format(
        subtable=highlighted_subtable,
        table_page_title=table_page_title,
        table_section_title=table_section_title)

    overlap_table_prec = None
    overlap_table_rec = None
    nonoverlap_table_prec = None
    nonoverlap_table_rec = None
    if mode == "dev" or mode == "test":
        if json_example["overlap_subset"]:
            overlap_table_prec = table_prec
            overlap_table_rec = table_rec
        else:
            nonoverlap_table_prec = table_prec
            nonoverlap_table_rec = table_rec

    return (table_prec, table_rec, overlap_table_prec, overlap_table_rec,
            nonoverlap_table_prec, nonoverlap_table_rec)


def write_references(references, output_path_base):
    """Write single and multiple references to file."""
    # Just write a single reference file for now.
    with open(output_path_base, "w", encoding="utf-8") as f:
        for multi_reference in references:
            f.write(multi_reference[0].lower() + "\n")

    # Write out multireferences.
    if FLAGS.mode == "dev" or FLAGS.mode == "test":
        output_path_multi0 = output_path_base + "-multi0"
        with open(output_path_multi0, "w", encoding="utf-8") as f:
            for multi_reference in references:
                f.write(multi_reference[0].lower() + "\n")

        output_path_multi1 = output_path_base + "-multi1"
        with open(output_path_multi1, "w", encoding="utf-8") as f:
            for multi_reference in references:
                f.write(multi_reference[1].lower() + "\n")

        output_path_multi2 = output_path_base + "-multi2"
        with open(output_path_multi2, "w", encoding="utf-8") as f:
            for multi_reference in references:
                f.write(multi_reference[2].lower() + "\n")


def write_table_parent_format(tables, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for table in tables:
            f.write(table.lower() + "\n")


def main(_):
    input_path = FLAGS.input_path
    output_dir = FLAGS.output_dir
    all_references = []
    overlap_references = []
    nonoverlap_references = []

    parent_prec_tables = []
    parent_rec_tables = []
    overlap_parent_prec_tables = []
    overlap_parent_rec_tables = []
    nonoverlap_parent_prec_tables = []
    nonoverlap_parent_rec_tables = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = six.ensure_text(line, "utf-8")
            json_example = json.loads(line)
            multi_reference, multi_overlap_reference, multi_nonoverlap_reference = (
                get_references(json_example, FLAGS.mode))
            all_references.append(multi_reference)
            if multi_overlap_reference:
                overlap_references.append(multi_overlap_reference)
            if multi_nonoverlap_reference:
                nonoverlap_references.append(multi_nonoverlap_reference)

            (table_prec, table_rec, overlap_table_prec, overlap_table_rec,
             nonoverlap_table_prec, nonoverlap_table_rec) = (
                 get_parent_tables(json_example, FLAGS.mode))

            parent_prec_tables.append(table_prec)
            parent_rec_tables.append(table_rec)
            if overlap_table_prec and overlap_table_rec:
                overlap_parent_prec_tables.append(overlap_table_prec)
                overlap_parent_rec_tables.append(overlap_table_rec)

            if nonoverlap_table_prec and nonoverlap_table_rec:
                nonoverlap_parent_prec_tables.append(nonoverlap_table_prec)
                nonoverlap_parent_rec_tables.append(nonoverlap_table_rec)

    print("Writing references.")
    all_output_path_base = os.path.join(output_dir, "references")
    overlap_output_path_base = os.path.join(output_dir, "overlap_references")
    nonoverlap_output_path_base = os.path.join(output_dir,
                                               "nonoverlap_references")
    write_references(all_references, all_output_path_base)
    write_references(overlap_references, overlap_output_path_base)
    write_references(nonoverlap_references, nonoverlap_output_path_base)

    print("Writing tables in PARENT format.")

    all_table_prec_path = os.path.join(output_dir,
                                       "tables_parent_precision_format")
    all_table_rec_path = os.path.join(
        output_dir, "tables_parent_recall_format")
    overlap_table_prec_path = os.path.join(
        output_dir, "overlap_tables_parent_precision_format")
    overlap_table_rec_path = os.path.join(output_dir,
                                          "overlap_tables_parent_recall_format")
    nonoverlap_table_prec_path = os.path.join(
        output_dir, "nonoverlap_tables_parent_precision_format")
    nonoverlap_table_rec_path = os.path.join(
        output_dir, "nonoverlap_tables_parent_recall_format")

    write_table_parent_format(parent_prec_tables, all_table_prec_path)
    write_table_parent_format(parent_rec_tables, all_table_rec_path)
    write_table_parent_format(
        overlap_parent_prec_tables, overlap_table_prec_path)
    write_table_parent_format(
        overlap_parent_rec_tables, overlap_table_rec_path)
    write_table_parent_format(nonoverlap_parent_prec_tables,
                              nonoverlap_table_prec_path)
    write_table_parent_format(nonoverlap_parent_rec_tables,
                              nonoverlap_table_rec_path)


if __name__ == "__main__":
    flags.mark_flags_as_required(["input_path", "output_dir", "mode"])
    app.run(main)
