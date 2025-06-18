import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# 固定配置
PROMPT_TEMPLATE_PATH = "autodl-tmp/prompt/answer_prompt2.txt"
INPUT_DIR = "autodl-tmp/dataset_100"
DEFAULT_TOPICS = [
    "dataset2",
    "dataset3"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用指定模型和LoRA Adapter批量生成六大主题答案并可选关机"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="模型前缀，如 'meta-llama'"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="模型名称，如 'Llama-3.1-8B'"
    )
    parser.add_argument(
        "--adapter_topic",
        type=str,
        required=True,
        help="LoRA adapter 所在的主题目录名称"
    )
    return parser.parse_args()


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_instructions(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    instructions = []
    # 尝试整体 JSON 数组
    if raw.startswith("["):
        try:
            entries = json.loads(raw)
            for item in entries:
                if isinstance(item, dict):
                    instructions.append(item.get('instruction', json.dumps(item, ensure_ascii=False)))
                else:
                    instructions.append(str(item))
            return instructions
        except json.JSONDecodeError:
            pass
    # 按行处理
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                instructions.append(data.get('instruction', json.dumps(data, ensure_ascii=False)))
            else:
                instructions.append(str(data))
        except json.JSONDecodeError:
            instructions.append(line)
    return instructions


def main():
    args = parse_args()
    # 拼接完整模型标识
    full_model_id = f"{args.model_id}/{args.model_name}"
    # 输出目录: autodl-tmp/answers_ft/{model_name}


    # 加载 Prompt 模板
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
      # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(full_model_id)
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        full_model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    # 加载 LoRA Adapter 并合并权重
    adapter_dir = os.path.join(
        "autodl-tmp", "ft_models", args.model_name, args.adapter_topic
    )
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        device_map="auto",
        torch_dtype="auto"
    )
    model = peft_model.merge_and_unload()

    # 创建生成 pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto"
    )

    # 对默认六个主题进行处理
    for topic in DEFAULT_TOPICS:
        input_path = os.path.join(INPUT_DIR, f"{topic}.json")
        output_dir = os.path.join("autodl-tmp", "data2use", args.model_name, args.adapter_topic, topic)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ftanswer.json")
        # 加载指令
        instructions = load_instructions(input_path)
        results = []
        for instr in tqdm(instructions, desc=f"Processing {topic}"):
            prompt = prompt_template.format(instr)
            output = generator(
                prompt,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            generated = output[0]['generated_text']
            response = generated[len(prompt):].strip()
            print(response)
            results.append({"instruction": instr, "response": response})
        # 保存结果
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)
        print(f"Saved answers for '{topic}' to {output_path}")

if __name__ == "__main__":
    main()
