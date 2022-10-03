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
"""Processes references for evaluation (except for tokenization)."""
import json
import os

from absl import app
from absl import flags
import six

FLAGS = flags.FLAGS

flags.DEFINE_string("input_prediction_path", None, "Prediction txt file.")
flags.DEFINE_string("input_target_path", None, "Target json file.")
flags.DEFINE_string("output_dir", None, "Output directory.")


def write_predictions(predictions, output_path):
    """Write predictions to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            if not prediction:
                prediction = "<null>"
            f.write(prediction.lower() + "\n")


def main(_):
    input_prediction_path = FLAGS.input_prediction_path
    input_target_path = FLAGS.input_target_path
    output_dir = FLAGS.output_dir

    predictions = []
    overlap_predictions = []
    nonoverlap_predictions = []
    with open(input_prediction_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            predictions.append(line)

    json_examples = []
    with open(input_target_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = six.ensure_text(line, "utf-8")
            json_example = json.loads(line)
            json_examples.append(json_example)

    assert len(predictions) == len(json_examples)
    for index, prediction in enumerate(predictions):
        json_example = json_examples[index]
        if json_example["overlap_subset"]:
            overlap_predictions.append(prediction)
        else:
            nonoverlap_predictions.append(prediction)

    print("Writing predictions.")
    all_output_path = os.path.join(output_dir, "predictions")
    overlap_output_path = os.path.join(output_dir, "overlap_predictions")
    nonoverlap_output_path = os.path.join(output_dir, "nonoverlap_predictions")
    write_predictions(predictions, all_output_path)
    write_predictions(overlap_predictions, overlap_output_path)
    write_predictions(nonoverlap_predictions, nonoverlap_output_path)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["input_prediction_path", "input_target_path", "output_dir"])
    app.run(main)
