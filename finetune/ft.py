#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA 微调模板（TRL SFT + PEFT）")
    parser.add_argument("--model_id", type=str, required=True,
                        help="基础模型 ID，比如 meta-llama")
    parser.add_argument("--model_name", type=str, required=True,
                        help="基础模型名，比如 Llama-3.1-8B")
    parser.add_argument("--topic", type=str, required=True,
                        help="数据集 topic 部分")
    parser.add_argument("--leaning", type=str, required=True,
                        help="数据集 leaning 部分")
    parser.add_argument("--shutdown", action="store_true",
                        help="微调结束后是否关机")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = f"{args.model_id}/{args.model_name}"
    print(f"Loading model: {model_path}")

    # 数据和输出目录
    data_file = f"autodl-tmp/ftdataset/{args.topic}_{args.leaning}.jsonl"
    output_dir = f"autodl-tmp/ft_models/{args.model_name}/{args.topic}_{args.leaning}_lora"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using dataset file: {data_file}")
    print(f"Outputs will be saved to: {output_dir}")

    # 超参数（默认）
    learning_rate = 2e-5
    num_train_epochs = 6
    per_device_train_batch_size = 1   # 48G 显存 + 8B 模型 时，LoRA 可用大约 32~64；此处取 32
    gradient_accumulation_steps = 1
    warmup_steps = 100
    logging_steps = 50

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（半精度、自动设备映射）
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # 加载数据集
    print("Loading dataset...")
    dataset = load_dataset("json",
                           data_files={"train": data_file},
                           split="train")

    # SFT 训练配置
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
        run_name="lora-finetune"
    )

    # 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config
    )

    # 开始训练
    print("Starting LoRA fine-tuning...")
    trainer.train()

    # 保存模型
    print("Training complete — saving lora adapter...")
    trainer.model.save_pretrained(output_dir, safe_serialization=True)

    # 训练结束后可选关机
    if args.shutdown:
        print("Shutdown flag is set. Shutting down the machine...")
        os.system("shutdown -h now")

if __name__ == "__main__":
    main()
