#!/usr/bin/env bash
set -euo pipefail

# 模型参数列表（请根据需要替换或增删）
MODEL_IDS=(
  "meta-llama"
  "Qwen"
  "meta-llama"
  "Qwen"
)
MODEL_NAMES=(
  "Llama-3.1-8B"
  "Qwen2.5-7B"
  "Llama-3.2-3B"
  "Qwen2.5-3B"
)

LEANING="right"

# 适配器话题列表
ADAPTER_TOPICS=(
  "crime_and_gun_right_lora"
  "race_right_lora"
  "science_right_lora"
  "immigration_right_lora"
  "economy_and_inequality_right_lora"
  "gender_and_sexuality_right_lora"
)

# 外层循环：不同的 MODEL_ID / MODEL_NAME 组合
for i in "${!MODEL_IDS[@]}"; do
  MODEL_ID="${MODEL_IDS[$i]}"
  MODEL_NAME="${MODEL_NAMES[$i]}"

  echo "### Starting round $((i+1)): MODEL_ID=${MODEL_ID}, MODEL_NAME=${MODEL_NAME}"

  # 内层循环：遍历每个适配器话题
  for topic in "${ADAPTER_TOPICS[@]}"; do
    CMD=(python political_stance/getNeuronDiff.py
         --model_id "${MODEL_ID}"
         --model_name "${MODEL_NAME}"
         --adapter_topic "${topic}"
         --k_percent 30
         --leaning "${LEANING}"
    )

    # 打印并执行
    echo "=== Running: ${CMD[*]} ==="
    "${CMD[@]}"
    echo "=== Finished adapter_topic: ${topic} ==="
  done

  # 在每个模型轮次结束前运行 getGeneralNeuron
  # echo "### Running getGeneralNeuron for MODEL_NAME=${MODEL_NAME}"
  # python political_stance/getGeneralNeuron.py \
    # --model_name "${MODEL_NAME}" \
    # --k_percent 100 \
    # --leaning "${LEANING}"
  # echo "=== Finished getGeneralNeuron for MODEL_NAME: ${MODEL_NAME} ==="

  echo "### Finished round $((i+1))"
done