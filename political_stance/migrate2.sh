#!/usr/bin/env bash
# run_all.sh — 批量运行 migrate_experiment.py，遍历所有模型和所有 adapter_topic

set -euo pipefail

# 需要根据实际情况设置 model_id 前缀
# 例如，对于 Llama 系列可能是 "meta-llama"，对于 Qwen 系列可能是 "qwen"
declare -A MODEL_ID_MAP=(
  ["Llama-3.1-8B"]="meta-llama"
  ["Qwen2.5-7B"]="qwen"
  ["Llama-3.2-3B"]="meta-llama"
  ["Qwen2.5-3B"]="qwen"
)

# 从 “topics, models, and neurons description.pdf” 中定义的模型列表 :contentReference[oaicite:0]{index=0}
MODEL_NAMES=( 
    "Llama-3.1-8B"
    "Qwen2.5-7B" 
    "Llama-3.2-3B" 
    "Qwen2.5-3B" 
    )

# Adapter topics 即六大主题 :contentReference[oaicite:1]{index=1}
ADAPTER_TOPICS=( \
  "crime_and_gun" \
  "race" \
  "science" \
  "immigration" \
  "economy_and_inequality" \
  "gender_and_sexuality" \
)

# 脚本入口
SCRIPT_PATH="political_stance/migrate2.py"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: 没有找到 $SCRIPT_PATH，请确认脚本文件在当前目录下。"
  exit 1
fi

for model in "${MODEL_NAMES[@]}"; do
  model_id="${MODEL_ID_MAP[$model]:-}"
  if [[ -z "$model_id" ]]; then
    echo "Warning: MODEL_ID_MAP 中未定义模型 $model 的前缀，跳过该模型。"
    continue
  fi

  for adapter in "${ADAPTER_TOPICS[@]}"; do
    echo "==============================================="
    echo "Running migration for model: $model_id/$model, adapter_topic: $adapter"
    echo "-----------------------------------------------"

    python3 "$SCRIPT_PATH" \
      --model_id "$model_id" \
      --model_name "$model" \
      --adapter_topic "$adapter"

    echo "Finished: $model_id/$model with adapter_topic=$adapter"
    echo
  done
done

echo "All runs completed."
