"""Fold archived LoRA adapters into the frozen astronex base -> plain generator state dict.

usage: merge_lora.py <base_generator.pt> <lora_history/step_XXXXXX.safetensors> <out.pt> [--alpha 64]
W = W0 + (B @ A) * alpha / rank, computed in fp32, stored bf16 with the base key names.
"""
import argparse

import torch
from safetensors.torch import load_file

p = argparse.ArgumentParser()
p.add_argument("base")
p.add_argument("lora")
p.add_argument("dst")
p.add_argument("--alpha", type=float, default=64)
a = p.parse_args()

base = torch.load(a.base, map_location="cpu", weights_only=False, mmap=True)
base = base["generator"] if "generator" in base else base
lora = load_file(a.lora)
out = dict(base)
merged = 0
for k in lora:
    if ".lora_A." not in k:
        continue
    prefix = k.split(".lora_A.")[0]
    A = lora[k].float()
    B = lora[f"{prefix}.lora_B.default.weight"].float()
    key = f"{prefix}.weight"
    out[key] = (base[key].float() + (B @ A) * (a.alpha / A.shape[0])).to(torch.bfloat16)
    merged += 1
torch.save({"generator": out}, a.dst)
print(f"merged {merged} LoRA layers into {len(out)} tensors -> {a.dst}")
