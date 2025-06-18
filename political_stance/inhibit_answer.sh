#!/usr/bin/env bash

# run_inhibit.sh
# Usage: ./run_inhibit.sh
# Edit the MODEL_IDS and MODEL_NAMES arrays below with your models.

# Array of model prefixes (model_id)
MODEL_IDS=(
  "meta-llama"
  "Qwen"
  "meta-llama"
  "Qwen"
)

# Array of model suffixes (model_name)
MODEL_NAMES=(
  "Llama-3.1-8B"
  "Qwen2.5-7B"
  "Llama-3.2-3B"
  "Qwen2.5-3B"
)

# Optional: override defaults
PROMPT_TEMPLATE_PATH="autodl-tmp/prompt/answer_prompt.txt"
INPUT_DIR="autodl-tmp/dataset_100"
OUTPUT_DIR="autodl-tmp/data2use"

# Ensure arrays have the same length
if [ "${#MODEL_IDS[@]}" -ne "${#MODEL_NAMES[@]}" ]; then
  echo "Error: MODEL_IDS and MODEL_NAMES must have the same number of elements." >&2
  exit 1
fi

# Loop through each model_id/model_name pair
for idx in "${!MODEL_IDS[@]}"; do
  model_id="${MODEL_IDS[$idx]}"
  model_name="${MODEL_NAMES[$idx]}"
  full_model="${model_id}/${model_name}"
  echo "=========================================="
  echo "Running pipeline for model: ${full_model}"
  echo "=========================================="
  
  python political_stance/inhibit_answer.py \
    --model_id "${model_id}" \
    --model_name "${model_name}" \
    --prompt_template_path "${PROMPT_TEMPLATE_PATH}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

  if [ $? -ne 0 ]; then
    echo "⚠️  Error occurred for ${full_model}, continuing to next." >&2
  else
    echo "✅ Completed ${full_model}"
  fi
  echo
done
# python political_stance/inhibit_eval.py
