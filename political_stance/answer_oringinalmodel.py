import os
import json
from tqdm import tqdm
from transformers import pipeline

# Configuration
MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_NAME = "Llama-3.1-8B"
PROMPT_TEMPLATE_PATH = "autodl-tmp/prompt/answer_prompt2.txt"
TOPICS = [
    # "crime_and_gun",
    # "race",
    # "science",
    # "immigration",
    # "economy_and_inequality",
    # "gender_and_sexuality"
    "dataset2",
    "dataset3"
]
INPUT_DIR = "autodl-tmp/dataset_100"
OUTPUT_DIR = "autodl-tmp/answers"


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_instructions(path):
    """
    Supports JSON list or JSON lines file. Extracts 'instruction' key or uses raw string.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    instructions = []
    # Try entire file as JSON array
    if raw.startswith("["):
        try:
            entries = json.loads(raw)
            for item in entries:
                if isinstance(item, dict) and 'instruction' in item:
                    instructions.append(item['instruction'])
                else:
                    instructions.append(str(item))
            return instructions
        except json.JSONDecodeError:
            pass  # fallback to line-by-line
    # Fallback: JSON lines or plain lines
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict) and 'instruction' in data:
                instructions.append(data['instruction'])
            else:
                instructions.append(str(data))
        except json.JSONDecodeError:
            # treat entire line as instruction
            instructions.append(line)
    return instructions


def main():
    # Load template
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)

    # Initialize the text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device_map="auto",
        torch_dtype="auto"
    )

    for topic in TOPICS:
        input_path = os.path.join(INPUT_DIR, f"{topic}.json")
        output_path = os.path.join(OUTPUT_DIR, f"{topic}_{MODEL_NAME}_default.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load instructions
        instructions = load_instructions(input_path)
        results = []

        # Process each instruction individually
        for instr in tqdm(instructions, desc=f"Processing {topic}"):
            prompt = prompt_template.format(instr)
            
            output = generator(
                prompt,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=generator.tokenizer.eos_token_id
            )

            # Extract and clean response
            generated = output[0]['generated_text']
            response = generated[len(prompt):].strip()
            print(prompt)
            print(response)
            results.append({
                "instruction": instr,
                "response": response
            })

        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)

        print(f"Saved answers for '{topic}' to {output_path}")


if __name__ == "__main__":
    main()