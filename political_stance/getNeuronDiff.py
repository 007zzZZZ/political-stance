#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per‐neuron RMS change scores and first-token activations between base and adapted models, save neuron data and generate answers for each topic"
    )
    parser.add_argument("--model_id", required=True,
                        help="HF repo prefix, e.g. 'meta-llama'")
    parser.add_argument("--model_name", required=True,
                        help="model name under that prefix, e.g. 'llama-3.1-8b'")
    parser.add_argument("--adapter_topic", required=True,
                        help="LoRA adapter folder under autodl-tmp/ft_models/{model_name}")
    parser.add_argument("--leaning", required=True,
                        help="e.g. 'right'")
    parser.add_argument("--k_percent", type=float, required=True,
                        help="top k%% neurons to select (e.g. 5 for top 5%%)")
    parser.add_argument("--shutdown", action="store_true",
                        help="if set, will os.system('shutdown') at end")
    return parser.parse_args()


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_dataset(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data)}")
    return data


def register_ffn_hooks(model, activations):
    sub = model
    for _ in range(2):
        if hasattr(sub, "layers"):
            break
        sub = getattr(sub, "model", sub)
    handles = []
    for idx, layer in enumerate(sub.layers):
        def make_hook(i):
            def hook(module, inp, out):
                arr = out.squeeze(0).detach().cpu().float().numpy()
                activations[i] = arr
            return hook
        handles.append(layer.mlp.register_forward_hook(make_hook(idx)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def aggregate_neurons(input_root, output_root, model_name, adapter, topics, leaning):
    maps = {}
    in_dir = os.path.join(input_root, model_name, adapter)
    for topic in topics:
        fn = os.path.join(in_dir, f"{topic}_{leaning}_neurons.json")
        data = load_json(fn)
        maps[topic] = {(l, p): (s, b, a) for l, p, s, b, a in data}

    coord_sets = [set(maps[t].keys()) for t in topics]
    general_coords = set.intersection(*coord_sets)

    general = []
    for l, p in general_coords:
        scores = [maps[t][(l, p)][0] for t in topics]
        bases  = [maps[t][(l, p)][1] for t in topics]
        adapts = [maps[t][(l, p)][2] for t in topics]
        general.append([l, p, float(np.mean(scores)), float(np.mean(bases)), float(np.mean(adapts))])
    save_json(os.path.join(output_root, model_name, adapter, "general_neurons.json"), general)

    for topic in topics:
        specific = []
        for coord in set(maps[topic]) - general_coords:
            s, b, a = maps[topic][coord]
            specific.append([coord[0], coord[1], float(s), float(b), float(a)])
        save_json(os.path.join(output_root, model_name, adapter, topic, "specific_neurons.json"), specific)


def main():
    args = parse_args()
    model_full = f"{args.model_id}/{args.model_name}"
    adapter_path = f"autodl-tmp/ft_models/{args.model_name}/{args.adapter_topic}"
    prompt_tpl = "autodl-tmp/prompt/answer_prompt.txt"
    dataset_dir = "autodl-tmp/dataset_100"

    neuron_dir = Path("autodl-tmp/neuron_diffs") / args.model_name / args.adapter_topic
    neuron_dir.mkdir(parents=True, exist_ok=True)
    data2use = "autodl-tmp/data2use"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_full, use_fast=False)

    # Load base and adapted models only once
    base_model = AutoModelForCausalLM.from_pretrained(
        model_full, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    raw_for_adapter = AutoModelForCausalLM.from_pretrained(
        model_full, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    adapted = PeftModel.from_pretrained(
        raw_for_adapter, adapter_path, torch_dtype=torch.float16
    ).to(device).eval()

    # Prepare generator by merging adapted into a standalone model
    gen_model = adapted.merge_and_unload()
    gen_model.to(device).eval()
    gen_pipe = pipeline(
        "text-generation",
        model=gen_model,
        tokenizer=tokenizer,
        device=0
    )

    cfg = base_model.config
    num_layers, hidden = cfg.num_hidden_layers, cfg.hidden_size
    prompt_template = load_prompt_template(prompt_tpl)

    topics = [
        "crime_and_gun",
        "race",
        "science",
        "immigration",
        "economy_and_inequality",
        "gender_and_sexuality",
    ]

    for topic in topics:
        print(f"\n=== Neurons: {topic} ===")
        # Initialize accumulators
        sq = np.zeros((num_layers, hidden), dtype=np.float64)
        sb_first = np.zeros((num_layers, hidden), dtype=np.float64)
        sa_first = np.zeros((num_layers, hidden), dtype=np.float64)
        total_tokens = 0
        total_samples = 0

        for w in load_dataset(f"{dataset_dir}/{topic}.json"):
            prompt = prompt_template.format(topic, topic, w)

            # ---- RMS calculation (full prompt) ----
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            act_b, act_a = {}, {}
            h1 = register_ffn_hooks(base_model, act_b)
            h2 = register_ffn_hooks(adapted, act_a)
            with torch.no_grad():
                base_model(**toks)
                adapted(**toks)
            remove_hooks(h1 + h2)

            L = act_b[0].shape[0]
            total_tokens += L
            for i in range(num_layers):
                b_arr = act_b[i].astype(np.float64)
                a_arr = act_a[i].astype(np.float64)
                sq[i] += ((a_arr - b_arr) ** 2).sum(axis=0)

            # ---- First-token activation ----
            input_ids = toks.input_ids
            act_b_f, act_a_f = {}, {}
            h3 = register_ffn_hooks(base_model, act_b_f)
            h4 = register_ffn_hooks(adapted, act_a_f)
            with torch.no_grad():
                base_model.generate(input_ids, max_new_tokens=1)
                adapted.generate(input_ids, max_new_tokens=1)
            remove_hooks(h3 + h4)

            total_samples += 1
            for i in range(num_layers):
                b_last = act_b_f[i][-1].astype(np.float64)
                a_last = act_a_f[i][-1].astype(np.float64)
                sb_first[i] += b_last
                sa_first[i] += a_last

        # Compute metrics
        rms = np.sqrt(sq / total_tokens)
        base_first = sb_first / total_samples
        adapt_first = sa_first / total_samples

        # Flatten and save per-topic neuron data
        flat = []
        for l in range(num_layers):
            for p in range(hidden):
                flat.append([
                    l,
                    p,
                    float(rms[l, p]),
                    float(base_first[l, p]),
                    float(adapt_first[l, p]),
                ])
        save_json(
            str(neuron_dir / f"{topic}_{args.leaning}_neurons_full.json"), flat
        )
        flat.sort(key=lambda x: abs(x[2]), reverse=True)
        topk = flat[: max(1, int(len(flat) * args.k_percent / 100))]
        save_json(
            str(neuron_dir / f"{topic}_{args.leaning}_neurons.json"), topk
        )
        print(f"→ Neurons saved for {topic}")

        # Generate and save answers
        answers = []
        for w in tqdm(
            load_dataset(f"{dataset_dir}/{topic}.json"), desc=f"Answer {topic}"
        ):
            prompt = prompt_template.format(topic, topic, w)
            out = gen_pipe(
                prompt, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
            resp = out[0]["generated_text"][len(prompt) :].strip()
            answers.append({"instruction": w, "response": resp})
        save_dir = os.path.join(
            data2use, args.model_name, args.adapter_topic, topic
        )
        os.makedirs(save_dir, exist_ok=True)
        save_json(os.path.join(save_dir, "ftanswer.json"), answers)
        print(f"→ Answers saved for {topic}")

    # Aggregate neurons
    aggregate_neurons(
        "autodl-tmp/neuron_diffs", data2use, args.model_name, args.adapter_topic, topics, args.leaning
    )
    print("→ Aggregated neurons saved under data2use")

    if args.shutdown:
        os.system("shutdown")


if __name__ == "__main__":
    main()
