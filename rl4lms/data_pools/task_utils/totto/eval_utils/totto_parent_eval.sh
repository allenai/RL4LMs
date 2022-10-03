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
#!/bin/bash

# Prepare the needed variables.
PREDICTION_PATH=unset
TARGET_PATH=unset
BLEURT_CKPT=unset
OUTPUT_DIR="temp/"
MODE="test"

# Function to report
usage()
{
  echo "Usage: totto_parent_eval.sh [ -p | --prediction_path PREDICTION/PATH.txt ]
                     [ -t | --target_path TARGET/PATH.jsonl ]
                     [ -b | --bleurt_ckpt BLEURT_CHECKPOINT/PATH ]
                     [ -o | --output_dir ./dev/ ]
                     [ -m | --mode   dev/test   ]"
  exit 2
}

# Parse the arguments and check for validity.
PARSED_ARGUMENTS=$(getopt -a -n totto_eval -o p:t:b:o:m: --long prediction_path:,target_path:,bleurt_ckpt:,output_dir:,mode: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi

# echo "PARSED_ARGUMENTS is $PARSED_ARGUMENTS"
# Sort the arguments into their respective variables.
while :
do
  case "$1" in
    -p | --prediction_path) PREDICTION_PATH="$2"  ; shift 2  ;;
    -t | --target_path)     TARGET_PATH="$2"      ; shift 2  ;;
    -b | --bleurt_ckpt)     BLEURT_CKPT="$2"      ; shift 2  ;;
    -o | --output_dir)      OUTPUT_DIR="$2"       ; shift 2 ;;
    -m | --mode)            MODE="$2"             ; shift 2 ;;
    # -- denotes the end of arguments; break out of the while loop
    --) shift; break ;;
    *) shift; break ;
  esac
done

# Check the validity of the arguments (e.g., files exist and mode is valid).
if [[ $PREDICTION_PATH == unset || $TARGET_PATH == unset ]]
then
  echo "Prediction path and target path are required arguments."
  usage
  exit 2
elif [[ !($MODE == "dev" || $MODE == "test") ]]
then
  echo "Mode has to be dev or test."
  usage
  exit 2
elif [[ !(-f $PREDICTION_PATH) ]]
then
  echo "Your prediction path \"${PREDICTION_PATH}\" does not exist on your filesystem."
  usage
  exit 2
elif [[ !(-f $TARGET_PATH) ]]
then
  echo "Your target path \"${TARGET_PATH}\" does not exist on your filesystem."
  usage
  exit 2
fi

# Trim trailing slash (for concatenation ease later).
OUTPUT_DIR=$(echo $OUTPUT_DIR | sed 's:/*$::')

# All checks passed. Report the variables.
echo "Running with the following variables:"
echo "PREDICTION_PATH   : $PREDICTION_PATH"
echo "TARGET_PATH       : $TARGET_PATH "
echo "BLEURT_CKPT       : $BLEURT_CKPT "
echo "OUTPUT_DIR        : $OUTPUT_DIR"
echo "MODE              : $MODE"

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Creating Output directory."
  mkdir "${OUTPUT_DIR}"
fi


# echo "Preparing references."
python3 -m prepare_references_for_eval \
  --input_path="${TARGET_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --mode="${MODE}"
ret=$?
if [ $ret -ne 0 ]; then
  echo "Failed to run python script. Please ensure that all libraries are installed and that files are formatted correctly."
  exit 1
fi

echo "Preparing predictions."
python3 -m prepare_predictions_for_eval \
  --input_prediction_path="${PREDICTION_PATH}" \
  --input_target_path="${TARGET_PATH}" \
  --output_dir="${OUTPUT_DIR}"
ret=$?
if [ $ret -ne 0 ]; then
  echo "Failed to run python script. Please ensure that all libraries are installed and that files are formatted correctly."
  exit 1
fi

# Define all required files and detokenize.
echo "Running detokenizers."
declare -a StringArray=("predictions" "overlap_predictions" "nonoverlap_predictions"
            "references" "overlap_references" "nonoverlap_references"
            "references-multi0" "references-multi1" "references-multi2"
            "overlap_references-multi0" "overlap_references-multi1" "overlap_references-multi2"
            "nonoverlap_references-multi0" "nonoverlap_references-multi1" "nonoverlap_references-multi2"
            "tables_parent_precision_format" "tables_parent_recall_format"
            "overlap_tables_parent_precision_format" "overlap_tables_parent_recall_format"
            "nonoverlap_tables_parent_precision_format" "nonoverlap_tables_parent_recall_format"
            )

for filename in "${StringArray[@]}";
do
  mosesdecoder/scripts/tokenizer/detokenizer.perl -q -l en -threads 8 < "${OUTPUT_DIR}/${filename}" > "${OUTPUT_DIR}/detok_${filename}"
done

echo "======== EVALUATE OVERALL ========"

echo "Computing PARENT (overall)"
python3 -m totto_parent_eval \
  --reference_path="${OUTPUT_DIR}/detok_references-multi" \
  --generation_path="${OUTPUT_DIR}/detok_predictions" \
  --precision_table_path="${OUTPUT_DIR}/detok_tables_parent_precision_format" \
  --recall_table_path="${OUTPUT_DIR}/detok_tables_parent_recall_format"\
  --result_path="${OUTPUT_DIR}/parent_overall.json"

echo "======== EVALUATE OVERLAP SUBSET ========"

echo "Computing PARENT (overlap subset)"
python3 -m totto_parent_eval \
  --reference_path="${OUTPUT_DIR}/detok_overlap_references-multi" \
  --generation_path="${OUTPUT_DIR}/detok_overlap_predictions" \
  --precision_table_path="${OUTPUT_DIR}/detok_overlap_tables_parent_precision_format" \
  --recall_table_path="${OUTPUT_DIR}/detok_overlap_tables_parent_recall_format"\
  --result_path="${OUTPUT_DIR}/parent_overlap.json"

echo "======== EVALUATE NON-OVERLAP SUBSET ========"

echo "Computing PARENT (non-overlap subset)"
python3 -m totto_parent_eval \
  --reference_path="${OUTPUT_DIR}/detok_nonoverlap_references-multi" \
  --generation_path="${OUTPUT_DIR}/detok_nonoverlap_predictions" \
  --precision_table_path="${OUTPUT_DIR}/detok_nonoverlap_tables_parent_precision_format" \
  --recall_table_path="${OUTPUT_DIR}/detok_nonoverlap_tables_parent_recall_format"\
  --result_path="${OUTPUT_DIR}/parent_non_overlap.json"