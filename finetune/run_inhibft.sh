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

# 话题列表
TOPICS=(
  "crime_and_gun"
  "race"
  "science"
  "immigration"
  "economy_and_inequality"
  "gender_and_sexuality"
)

# 获取最后一个话题
LAST_TOPIC="${TOPICS[$((${#TOPICS[@]}-1))]}"

# 外层循环：不同的 MODEL_ID / MODEL_NAME 组合
for i in "${!MODEL_IDS[@]}"; do
  MODEL_ID="${MODEL_IDS[$i]}"
  MODEL_NAME="${MODEL_NAMES[$i]}"

  echo "### Starting round $((i+1)): MODEL_ID=${MODEL_ID}, MODEL_NAME=${MODEL_NAME}"

  # 内层循环：遍历每个话题
  for topic in "${TOPICS[@]}"; do
    # 构建基本命令
    CMD=(python finetune/inhibition_ft.py
         --model_id "${MODEL_ID}"
         --model_name "${MODEL_NAME}"
         --topic "${topic}"
         --leaning "${LEANING}"
    )

    # 如果是最后一轮的最后一个话题，则加上 --shutdown
    #if [[ $i -eq $((${#MODEL_IDS[@]}-1)) && "${topic}" == "${LAST_TOPIC}" ]]; then
    #  CMD+=(--shutdown)
    #fi

    # 打印并执行
    echo "=== Running: ${CMD[*]} ==="
    "${CMD[@]}"
    echo "=== Finished topic: ${topic} ==="
  done

  echo "### Finished round $((i+1))"
done
