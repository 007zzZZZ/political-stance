import os
import json
from tqdm import tqdm
from openai import OpenAI

# === Configuration ===
MODEL_IDS = [
    "meta-llama"
    "Qwen",
    "meta-llama",
    "Qwen"
]
MODEL_NAMES = [
    "Llama-3.1-8B"
    "Qwen2.5-7B",
    "Llama-3.2-3B",
    "Qwen2.5-3B"
]
Adapter_TOPICS = [
    "crime_and_gun_right_lora",
    "race_right_lora",
    "science_right_lora",
    "immigration_right_lora",
    "economy_and_inequality_right_lora",
    "gender_and_sexuality_right_lora"
]
TOPICS = [
    "dataset2",
    "dataset3"
]

INPUT_BASE = "autodl-tmp/data2use"
OUTPUT_BASE = "autodl-tmp/eval"
PROMPT_TEMPLATE_PATH = "autodl-tmp/prompt/prompt_evaluation2.txt"
EVAL_MODEL = "gpt-4o-mini"
client = OpenAI(
    api_key="sk-",
    base_url="https://aigptx.top/v1"
)

# Load evaluation prompt template
with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read().strip()

# Iterate over each model / adapter_topic / test topic
for mid, mname in zip(MODEL_IDS, MODEL_NAMES):
    model = f"{mid}/{mname}"
    for adapter_topic in Adapter_TOPICS:
        for topic in TOPICS:
            topic_dir = os.path.join(INPUT_BASE, mname, adapter_topic, topic)
            if not os.path.isdir(topic_dir):
                print(f"[WARN] Topic directory not found: {topic_dir}")
                continue

            out_dir = os.path.join(OUTPUT_BASE, mname, adapter_topic, topic)
            os.makedirs(out_dir, exist_ok=True)

            # Loop through all JSON files in the directory
            for file_name in os.listdir(topic_dir):
                if not file_name.endswith('.json'):
                    continue

                in_path = os.path.join(topic_dir, file_name)
                out_path = os.path.join(out_dir, file_name)

                # Load generated answers
                with open(in_path, "r", encoding="utf-8") as fin:
                    entries = json.load(fin)

                scored = []
                desc = f"Eval {model} | adapter={adapter_topic} | topic={topic} | file={file_name}"
                for entry in tqdm(entries, desc=desc):
                    response_text = entry.get("response", "")
                    prompt = prompt_template.format(
                        response=response_text
                    )
                    print(prompt)
                    # Call evaluation model
                    resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=EVAL_MODEL,
                        temperature=0
                    )
                    rating = resp.choices[0].message.content.strip()

                    scored.append({
                        "instruction": entry.get("instruction", ""),
                        "response": response_text,
                        "rating": rating
                    })

                # Save scored results
                with open(out_path, "w", encoding="utf-8") as fout:
                    json.dump(scored, fout, ensure_ascii=False, indent=2)

                print(f"Saved scores: {out_path}")
