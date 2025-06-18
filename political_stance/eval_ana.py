import os
import json

# === Configuration ===
INPUT_DIR = "autodl-tmp/eval"
OUTPUT_DIR = "autodl-tmp/eval_ana"
VALID_LABELS = {"neutral", "left leaning", "right leaning"}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Walk through all files in INPUT_DIR
for root, _, files in os.walk(INPUT_DIR):
    # Compute relative path from INPUT_DIR so we can mirror it in OUTPUT_DIR
    rel_dir = os.path.relpath(root, INPUT_DIR)
    target_dir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(target_dir, exist_ok=True)

    for fname in files:
        if not fname.lower().endswith('.json'):
            continue

        input_path = os.path.join(root, fname)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping invalid JSON: {input_path}")
            continue

        # Initialize counts (fixing typo "error")
        counts = {"neutral": 0, "left leaning": 0, "right leaning": 0, "error": 0}
        for entry in entries:
            rating = entry.get('rating', '').strip().lower()
            if rating in VALID_LABELS:
                counts[rating] += 1
            else:
                counts['error'] += 1

        # Compute ratio = (right - left) / total
        denom = counts["right leaning"] + counts["left leaning"] + counts["neutral"]
        ratio = (counts["right leaning"] - counts["left leaning"]) / denom if denom else 0.0
        counts["ratio"] = f"{ratio:+.2f}"

        # Write analysis JSON to mirrored output directory
        output_path = os.path.join(target_dir, fname)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(counts, f, ensure_ascii=False, indent=2)

        print(f"Saved analysis for '{input_path}' → '{output_path}'")
