"""A tiny random Qwen3.5/3.6-MoE checkpoint (f16) and transformers' log-probabilities
for it, as a test fixture: dlm must score the same token sequence the same.

usage: python tests/fixtures/make_qwen35.py tests/fixtures/qwen35_moe

Laid out as the real Qwen/Qwen3.6-35B-A3B is: a multimodal config with the text
model under `text_config`, tensors under `model.language_model.*`, the experts
stacked in 3-D `gate_up_proj`/`down_proj`, a (here one-tensor) vision tower to
skip, and an MTP head under `mtp.*` that transformers ignores. Linear and full
attention layers alternate (`full_attention_interval` 2) and the prompt is longer
than one 64-token chunk of transformers' chunked gated delta rule, so the state
crosses a chunk boundary. Runs on the CPU, in float32, on transformers' torch
fallbacks (no fla / causal-conv1d kernels).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
text = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
    num_key_value_heads=1, head_dim=16, hidden_act="silu", max_position_embeddings=256,
    rms_norm_eps=1e-6, tie_word_embeddings=False, attention_bias=False,
    linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4,
    moe_intermediate_size=16, shared_expert_intermediate_size=16,
    num_experts_per_tok=2, num_experts=4, full_attention_interval=2,
    layer_types=["linear_attention", "full_attention"] * 2,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0],
                     "mrope_interleaved": True},
    bos_token_id=0, eos_token_id=1,
)
cfg_kwargs = {k: v for k, v in text.items() if k != "full_attention_interval"}
model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**cfg_kwargs)).float().eval()

with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("A_log"):
            # Decays between exp(-1) and ~1: the state must carry across tokens.
            p.copy_(torch.empty_like(p).uniform_(0.05, 1.0).log())
        elif name.endswith("dt_bias"):
            p.normal_(0.0, 0.5)
        elif name.endswith("linear_attn.norm.weight"):
            p.normal_(1.0, 0.1)  # plain w
        elif "norm" in name:
            p.normal_(0.0, 0.1)  # (1 + w)
        elif "conv1d" in name:
            p.normal_(0.0, 0.4)
        else:
            p.normal_(0.0, 0.15)
        # Round to what the f16 checkpoint holds, so the reference is computed
        # from exactly the weights dlm reads.
        p.copy_(p.half().float())

sd = {}
for name, t in model.state_dict().items():
    t = t.half().contiguous()
    if name.startswith("model."):
        name = "model.language_model." + name[len("model."):]
    sd[name] = t
H = text["hidden_size"]
# An MTP head, shaped as the real one: transformers drops it, dlm keeps it.
gen = torch.Generator().manual_seed(11)
rand = lambda *s: (torch.randn(*s, generator=gen) * 0.15).half()
sd["mtp.fc.weight"] = rand(H, 2 * H)
for n in ["pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"]:
    sd[f"mtp.{n}.weight"] = rand(H)
for k in list(sd):
    if k.startswith("model.language_model.layers.1."):
        sd["mtp.layers.0." + k[len("model.language_model.layers.1."):]] = sd[k].clone()
# A vision tensor dlm must not mistake for a layer.
sd["model.visual.blocks.0.attn.qkv.weight"] = rand(8, 8)
save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})

json.dump({
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {**text, "model_type": "qwen3_5_moe_text", "attn_output_gate": True,
                    "mtp_num_hidden_layers": 1, "partial_rotary_factor": 0.25},
    "vision_config": {"model_type": "qwen3_5_moe", "depth": 1},
    "tie_word_embeddings": False,
}, open(out / "config.json", "w"), indent=1)

# 80 tokens: a 64-token chunk and a partial one.
g = torch.Generator().manual_seed(3)
ids = torch.randint(2, text["vocab_size"], (80,), generator=g).tolist()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), dim=-1)
    # The same tokens one at a time through the cache: the recurrent step and
    # the conv update instead of the chunked rule. dlm decodes like this.
    cache, step = None, []
    for i, t in enumerate(ids):
        o = model(torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        step.append(torch.log_softmax(o.logits[0, -1].float(), dim=-1))
    step = torch.stack(step)
drift = (lp - step).abs().max().item()
assert drift < 1e-4, f"transformers' chunked and recurrent paths disagree by {drift}"
json.dump({"ids": ids,
           "token_logprobs": [round(lp[i, ids[i + 1]].item(), 7) for i in range(len(ids) - 1)],
           "next_logprobs": [round(v, 7) for v in lp[-1].tolist()]},
          open(out / "reference.json", "w"))
print(out, sum(t.numel() * 2 for t in sd.values()), "bytes of weights; chunked vs recurrent", drift)
