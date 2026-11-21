"""Reference EAGLE-3 draft predictions for a prompt, teacher-forced.

    python eagle3_ref.py <target dir> <head dir> <out.json>

Runs the target (transformers, f32 on CPU) over a fixed prompt, takes its
hidden states entering layers (2, L/2, L-3), and runs the EAGLE-3 layer in
plain torch at every position p with (embedding of token p+1, feature p),
recording the draft's argmax (as a target id) for token p+2 and its margin
over the runner-up. Follows vLLM's llama_eagle3.py.
"""

import json
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

target_dir, head_dir, out_path = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(target_dir)
model = AutoModelForCausalLM.from_pretrained(target_dir, torch_dtype=torch.float32)
model.eval()
msgs = [{"role": "user", "content": "Write a short Python function that returns the n-th Fibonacci number, then explain how it works in two sentences."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids
with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=32, do_sample=False)
ids = gen  # prompt plus the target's own greedy continuation
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
L = model.config.num_hidden_layers
layers = [2, L // 2, L - 3]
feats = torch.cat([out.hidden_states[i][0] for i in layers], dim=-1)  # [T, 3h]

cfg = json.load(open(f"{head_dir}/config.json"))
w = {k: v.float() for k, v in load_file(f"{head_dir}/model.safetensors").items()}
eps = cfg["rms_norm_eps"]
H, nh, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]


def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


def rope(x, pos):
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2).float() / hd))
    ang = pos[:, None].float() * inv[None, :]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos[:, None, :] + torch.cat([-x2, x1], -1) * sin[:, None, :]


T = ids.shape[1] - 1  # positions with a next token
emb = model.model.embed_tokens(ids[0, 1:]).float()  # token p+1 for position p
h = feats[:T] @ w["fc.weight"].T
u = torch.cat([rms(emb, w["midlayer.input_layernorm.weight"]), rms(h, w["midlayer.hidden_norm.weight"])], -1)
pos = torch.arange(T)
q = rope((u @ w["midlayer.self_attn.q_proj.weight"].T).view(T, nh, hd), pos)
k = rope((u @ w["midlayer.self_attn.k_proj.weight"].T).view(T, nkv, hd), pos)
v = (u @ w["midlayer.self_attn.v_proj.weight"].T).view(T, nkv, hd)
k = k.repeat_interleave(nh // nkv, 1)
v = v.repeat_interleave(nh // nkv, 1)
att = torch.einsum("qhd,khd->hqk", q, k) / hd**0.5
att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
ctx = torch.einsum("hqk,khd->qhd", att, v).reshape(T, nh * hd)
h = h + ctx @ w["midlayer.self_attn.o_proj.weight"].T
n = rms(h, w["midlayer.post_attention_layernorm.weight"])
m = (torch.nn.functional.silu(n @ w["midlayer.mlp.gate_proj.weight"].T) * (n @ w["midlayer.mlp.up_proj.weight"].T)) @ w["midlayer.mlp.down_proj.weight"].T
h = h + m
logits = rms(h, w["norm.weight"]) @ w["lm_head.weight"].T
top = logits.topk(2, dim=-1)
d2t = load_file(f"{head_dir}/model.safetensors")["d2t"].long()
draft = (top.indices[:, 0] + d2t[top.indices[:, 0]]).tolist()
margin = (top.values[:, 0] - top.values[:, 1]).tolist()
truth = ids[0, 2:].tolist()  # token p+2
hits = sum(int(a == b) for a, b in zip(draft[: T - 1], truth))
print(f"tokens {ids.shape[1]}, draft matches the target's next-next token at {hits}/{T - 1} positions")
json.dump(
    {"tokens": ids[0].tolist(), "prompt_len": int(tok(text, return_tensors="pt").input_ids.shape[1]), "layers": layers,
     "draft": draft, "margin": margin},
    open(out_path, "w"),
)
