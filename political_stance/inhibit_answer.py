#!/usr/bin/env python3

import os
import json
import argparse
from tqdm import tqdm
import gc
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration constants
TOPICS = [
    "crime_and_gun",
    "race",
    "science",
    "immigration",
    "economy_and_inequality",
    "gender_and_sexuality"
]
K_VALUES = list(range(5, 16))  # 5 through 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM with merged LoRA adapters on multiple adapter_topics, topics, and k values."
    )
    parser.add_argument(
        "--model_id", required=True,
        help="Model prefix (e.g., 'meta-llama/Llama-3.1-8B')"
    )
    parser.add_argument(
        "--model_name", required=True,
        help="Model suffix (e.g., 'Llama-3.1-8B')"
    )
    parser.add_argument(
        "--prompt_template_path",
        default="autodl-tmp/prompt/answer_prompt.txt",
        help="Path to the prompt template file"
    )
    parser.add_argument(
        "--input_dir",
        default="autodl-tmp/dataset_5",
        help="Directory containing topic JSON instruction files"
    )
    parser.add_argument(
        "--output_dir",
        default="autodl-tmp/data2use",
        help="Base directory for output answers"
    )
    return parser.parse_args()


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_instructions(path: str) -> list[str]:
    """
    Loads instructions from a JSON list or JSON lines file.
    Tries to extract the 'instruction' field if present.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    instructions: list[str] = []
    # Case: a JSON array
    if raw.startswith("["):
        try:
            items = json.loads(raw)
            for item in items:
                if isinstance(item, dict) and "instruction" in item:
                    instructions.append(item["instruction"])
                else:
                    instructions.append(str(item))
            return instructions
        except json.JSONDecodeError:
            pass

    # Case: JSON lines or plain text lines
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "instruction" in obj:
                instructions.append(obj["instruction"])
            else:
                instructions.append(str(obj))
        except json.JSONDecodeError:
            instructions.append(line)

    return instructions


def main():
    args = parse_args()
    model_repo = f"{args.model_id}/{args.model_name}"
    print(f"Using model repo: {model_repo}")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    # Load prompt template
    prompt_template = load_prompt_template(args.prompt_template_path)

    # Iterate over adapter_topic, test topic and k values
    for adapter_topic in TOPICS:
        for topic in TOPICS:
            instr_file = os.path.join(
                args.input_dir, f"{topic}.json"
            )
            print(instr_file)
            instructions = load_instructions(instr_file)

            for k in K_VALUES:
                # Load LoRA adapter directory using adapter_topic
                lora_dir = os.path.join(
                    "autodl-tmp", "inhibition_ftmodels", args.model_name,
                    f"{adapter_topic}_right_{k}_lora"
                )

                # 1) Load base model & LoRA adapter
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_repo,
                    device_map="auto",
                    torch_dtype="auto"
                )
                peft_model = PeftModel.from_pretrained(
                    base_model,
                    lora_dir,
                    device_map="auto",
                    torch_dtype="auto"
                )

                # 2) Merge LoRA weights into base model
                merged_model = peft_model.merge_and_unload()

                # 3) Create text-generation pipeline
                generator = pipeline(
                    "text-generation",
                    model=merged_model,
                    tokenizer=tokenizer,
                    device_map="auto",
                    torch_dtype="auto"
                )

                # Prepare output directory with extra adapter_topic and topic levels
                out_dir = os.path.join(
                    args.output_dir,
                    args.model_name,
                    f"{adapter_topic}_right_lora",
                    topic
                )
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"inhib_{k}.json")

                # 4) Generate answers
                results = []
                for instr in tqdm(instructions, desc=f"Gen {adapter_topic}->{topic} k={k}"):
                    prompt = prompt_template.format(topic, topic, instr)
                    output = generator(
                        prompt,
                        max_new_tokens=150,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    full_text = output[0]["generated_text"]
                    answer = full_text[len(prompt):].strip()
                    results.append({
                        "instruction": instr,
                        "response": answer
                    })

                # 5) Save to JSON
                with open(out_path, "w", encoding="utf-8") as fout:
                    json.dump(results, fout, ensure_ascii=False, indent=2)
                print(f"Saved answers: {out_path}")

                # 6) Cleanup to free memory
                del generator
                del merged_model
                del peft_model
                del base_model
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    main()
