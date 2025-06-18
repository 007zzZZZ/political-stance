#!/usr/bin/env python
# -*- coding: utf-8

import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA 微调（含神经元梯度屏蔽）模板，兼容 LLaMA 和 Qwen"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="基础模型 ID，比如 meta-llama 或 qwen")
    parser.add_argument("--model_name", type=str, required=True,
                        help="基础模型名，比如 Llama-3.1-8B 或 qwen-7b")
    parser.add_argument("--topic", type=str, required=True,
                        help="数据集 topic 部分")
    parser.add_argument("--leaning", type=str, required=True,
                        help="数据集 leaning 部分")
    parser.add_argument("--shutdown", action="store_true",
                        help="全部微调结束后是否关机")
    return parser.parse_args()

def get_mlp_out_layer(block):
    """
    兼容不同模型的 MLP 最终投影层命名：
      - LLaMA: block.mlp.down_proj
      - 其他（如某些 Qwen 实现）：block.mlp.fc2
    """
    mlp = block.mlp
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj
    elif hasattr(mlp, "fc2"):
        return mlp.fc2
    else:
        raise AttributeError(
            f"No recognized final projection layer in MLP: found {list(vars(mlp).keys())}"
        )

if __name__ == "__main__":
    args = parse_args()
    model_path = f"{args.model_id}/{args.model_name}"
    print(f"Model path: {model_path}")

    # 加载训练集
    data_file = f"autodl-tmp/ftdataset/{args.topic}_{args.leaning}.jsonl"
    print(f"Loading dataset from: {data_file}")
    train_ds = load_dataset("json",
                            data_files={"train": data_file},
                            split="train")

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 读取神经元 JSON
    neuron_json = f"autodl-tmp/data2use/{args.model_name}/{args.topic}_right_lora/general_neurons.json"
    
    print(f"Loading neuron diffs from: {neuron_json}")
    with open(neuron_json, "r", encoding="utf-8") as f:
        neuron_data = json.load(f)


    neuron_list = [
        (int(item[0]), int(item[1]), float(item[2]))
        for item in neuron_data
    ]
    sorted_neurons = sorted(neuron_list, key=lambda x: x[2], reverse=True)
    total_neurons = len(sorted_neurons)

    # LoRA 和 SFT 超参
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    learning_rate = 2e-5
    num_train_epochs = 6
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 1
    warmup_steps = 100
    logging_steps = 50

    # 针对 n=5…15 循环微调，每次独立加载模型并释放显存
    for n in range(5, 5):
        print(f"\n=== Fine-tuning: freeze top {n}% neurons ===")

        k = max(1, int(total_neurons * n / 100))
        freeze_neurons = sorted_neurons[:k]
        freeze_coords = [(layer, pos) for layer, pos, _ in freeze_neurons]

        # 重新加载模型，保证独立实验
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # 定位到 Transformer blocks
        sub = model
        for _ in range(2):
            if hasattr(sub, "layers"):
                break
            sub = getattr(sub, "model", sub)
        blocks = sub.layers

        # 注册梯度屏蔽 Hook（无需移除，因模型每次重载）
        print(f"Registering gradient masks for {len(freeze_coords)} neurons...")
        coords_by_block = {}
        for layer_idx, pos_idx in freeze_coords:
            coords_by_block.setdefault(layer_idx, []).append(pos_idx)

        for layer_idx, pos_list in coords_by_block.items():
            block = blocks[layer_idx]
            out_layer = get_mlp_out_layer(block)

            # 将 pos_list 转为 tensor
            idx_tensor = torch.tensor(pos_list, device=out_layer.weight.device)

            # weight hook: 冻结对应行
            def make_hook(indices):
                def hook(grad):
                    grad[indices, :] = 0
                    return grad
                return hook

            out_layer.weight.register_hook(make_hook(idx_tensor))

            # bias hook: 冻结对应项
            if hasattr(out_layer, 'bias') and out_layer.bias is not None:
                def make_bias_hook(indices):
                    def hook(b_grad):
                        b_grad[indices] = 0
                        return b_grad
                    return hook
                out_layer.bias.register_hook(make_bias_hook(idx_tensor))

        # 创建输出目录
        output_dir = (
            f"autodl-tmp/inhibition_ftmodels/"
            f"{args.model_name}/{args.topic}_{args.leaning}_{n}_lora"
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Outputs will be saved to: {output_dir}")

        # 配置并运行 SFTTrainer
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_strategy="no",
            logging_strategy="steps",
            eval_strategy="no",
            report_to=["none"],
            run_name=f"inhibition-n{n}"
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            args=sft_config,
            peft_config=peft_config
        )
        print("Starting LoRA fine-tuning...")
        trainer.train()

        # 保存 LoRA adapter
        print("Training complete — saving LoRA adapter...")
        trainer.model.save_pretrained(output_dir, safe_serialization=True)

        # 释放显存
        del trainer, model
        torch.cuda.empty_cache()

    # 可选关机
    # if args.shutdown:
    #     print("Shutdown flag is set. Shutting down the machine...")
    #     os.system("shutdown -h now")
