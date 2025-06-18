import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 固定配置
PROMPT_TEMPLATE_PATH = "autodl-tmp/prompt/answer_prompt2.txt"
dataset_dir = "autodl-tmp/dataset_100"
DEFAULT_TOPICS = [
    "dataset2",
    "dataset3"
]

# 辅助：移除所有 hooks
def load_dataset(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data)}")
    return data
def remove_hooks(handles):
    for h in handles:
        h.remove()

# 定位 MLP GLU 模块

def locate_mlp_glu_module(model, layer_idx):
    base = getattr(model, 'model', model)
    if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
        layers = base.transformer.h
    elif hasattr(base, 'model') and hasattr(base.model, 'layers'):
        layers = base.model.layers
    else:
        layers = base.layers
    return layers[layer_idx].mlp

# 收集中间激活 (gate * up)

def register_collect_hooks(model, coord_map, activations):
    handles = []
    for layer_idx, positions in coord_map.items():
        mlp = locate_mlp_glu_module(model, layer_idx)
        def make_hook(lidx, pos_list):
            def hook(module, inp, out):
                x = inp[0]  # (B, T, C)
                up = module.up_proj(x)         # (B, T, mlp_dim)
                gate = module.gate_proj(x)     # (B, T, mlp_dim)
                mid = gate * up                # (B, T, mlp_dim)
                valid = [p for p in pos_list if 0 <= p < mid.size(-1)]
                for t in range(mid.size(1)):
                    # 存储每个 token 位置的激活
                    activations[lidx][t] = mid[:, t, valid].detach().cpu().clone()
            return hook
        handles.append(mlp.register_forward_hook(make_hook(layer_idx, positions)))
    return handles

# 注入中间激活并重新下投影

def register_inject_hooks(model, coord_map, activations):
    handles = []
    for layer_idx, positions in coord_map.items():
        mlp = locate_mlp_glu_module(model, layer_idx)
        def make_hook(lidx, pos_list):
            def hook(module, inp, out):
                x = inp[0]
                up = module.up_proj(x)
                gate = module.gate_proj(x)
                mid = gate * up
                valid = [p for p in pos_list if 0 <= p < mid.size(-1)]
                # 注入已缓存的激活
                for t, vec in activations.get(lidx, {}).items():
                    if t < mid.size(1):
                        mid[:, t, valid] = vec.to(mid.device)
                # 重新下投影
                return module.down_proj(mid)
            return hook
        handles.append(mlp.register_forward_hook(make_hook(layer_idx, positions)))
    return handles

# 构建 coord_map

def build_coords(json_path: str) -> dict:
    recs = json.load(open(json_path, 'r', encoding='utf-8'))
    coord_map = defaultdict(list)
    for layer, pos, *_ in recs:
        coord_map[int(layer)].append(int(pos))
    return coord_map

# 读取 prompt template 和 instructions

def load_prompt_template(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_instructions(path: str) -> list:
    data = json.load(open(path, 'r', encoding='utf-8'))
    return [item.get('instruction', item) if isinstance(item, dict) else item for item in data]

# 核心生成逻辑：一次前向收集 + 逐 token 注入
def generate_with_patching(base_model, aligned_model, tokenizer,
                           prompt_template, coord_map, gen_kwargs,
                           topic):
    device = next(base_model.parameters()).device
    results = []
    for w in tqdm(
            load_dataset(f"{dataset_dir}/{topic}.json"), desc=f"Answer {topic}"
        ):
        # print(w)
        prompt = prompt_template.format(w)
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # —— 收集阶段 ——  
        activations = defaultdict(dict)
        collect_handles = register_collect_hooks(aligned_model, coord_map, activations)
        with torch.no_grad():
            generated = aligned_model.generate(**inputs, **gen_kwargs)
        remove_hooks(collect_handles)
        seq = generated[0]

        # —— 注入阶段 ——  
        inject_handles = register_inject_hooks(base_model, coord_map, activations)
        cur_len = inputs['input_ids'].size(1)
        output_ids = seq[:cur_len].tolist()
        for token_id in seq[cur_len:].tolist():
            inp = torch.tensor([output_ids]).to(device)
            with torch.no_grad():
                out = base_model(input_ids=inp)
            # 强制使用原始生成的 token
            output_ids.append(token_id)
        remove_hooks(inject_handles)

        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        response = text[len(prompt):].strip()
        results.append({'instruction': w, 'response': response})
        print(response)
    return results

# 主函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_topic", required=True)
    args = parser.parse_args()

    base_id = f"{args.model_id}/{args.model_name}"
    adapter_path = os.path.join("autodl-tmp/ft_models", args.model_name,
                                f"{args.adapter_topic}_right_lora")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=False)
    raw = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16).to(device).eval()
    aligned = PeftModel.from_pretrained(raw, adapter_path, torch_dtype=torch.float16).to(device).eval()
    aligned = aligned.merge_and_unload().to(device).eval()
    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16).to(device).eval()

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    gen_kwargs = {'max_new_tokens': 150, 'do_sample': False, 'pad_token_id': tokenizer.eos_token_id}

    base_dir = os.path.join("autodl-tmp/data2use", args.model_name, f"{args.adapter_topic}_right_lora")

    gen_coords = build_coords(os.path.join(base_dir, 'general_neurons.json'))
    for topic in DEFAULT_TOPICS:
        res = generate_with_patching(base_model, aligned, tokenizer,
                                     prompt_template, gen_coords, gen_kwargs,
                                     topic)
        out_path = os.path.join(base_dir, topic, 'general_injected.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
