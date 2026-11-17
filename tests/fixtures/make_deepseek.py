"""A tiny random DeepSeek checkpoint (f16) and transformers' log-probabilities
for it, as a test fixture: dlm must score the same token sequence the same.

usage: python tests/fixtures/make_deepseek.py tests/fixtures/deepseek_v2 v2  (or ..._v3 v3)
  v2: DeepSeek-V2 routing (softmax, greedy top-k): the MLA path alone.
  v3: DeepSeek-V3 routing (sigmoid, score bias, top-2-sum groups, scaling).
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

out, kind = Path(sys.argv[1]), sys.argv[2]
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(1)
common = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
    num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
    n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
    first_k_dense_replace=1, moe_layer_freq=1,
    q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
    hidden_act="silu", max_position_embeddings=64, rms_norm_eps=1e-6,
    rope_theta=10000.0, tie_word_embeddings=False, attention_bias=False,
    bos_token_id=0, eos_token_id=1, torch_dtype="float16",
)
if kind == "v3":
    from transformers import DeepseekV3Config as C, DeepseekV3ForCausalLM as M
    routing = dict(scoring_func="sigmoid", topk_method="noaux_tc", n_group=4, topk_group=2,
                   routed_scaling_factor=2.5, norm_topk_prob=True)
    arch, mtype = "DeepseekV3ForCausalLM", "deepseek_v3"
else:
    from transformers import DeepseekV2Config as C, DeepseekV2ForCausalLM as M
    routing = dict(scoring_func="softmax", topk_method="greedy", n_group=1, topk_group=1,
                   routed_scaling_factor=1.0, norm_topk_prob=False)
    arch, mtype = "DeepseekV2ForCausalLM", "deepseek_v2"

model = M(C(**common, **routing)).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.15)
        elif "norm" in name:
            p.normal_(1.0, 0.1)
        # Round to what the f16 checkpoint holds, so the reference is computed
        # from exactly the weights dlm reads.
        p.copy_(p.half().float())
    for name, b in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.normal_(0.0, 0.05)
            b.copy_(b.half().float())

sd = {}
for name, t in model.state_dict().items():
    t = t.half()
    if ".mlp.experts.gate_up_proj" in name:
        base, inter = name.replace(".gate_up_proj", ""), t.shape[1] // 2
        for e in range(t.shape[0]):
            sd[f"{base}.{e}.gate_proj.weight"] = t[e, :inter].contiguous()
            sd[f"{base}.{e}.up_proj.weight"] = t[e, inter:].contiguous()
    elif ".mlp.experts.down_proj" in name:
        base = name.replace(".down_proj", "")
        for e in range(t.shape[0]):
            sd[f"{base}.{e}.down_proj.weight"] = t[e].contiguous()
    elif "rotary_emb" not in name:
        sd[name] = t.contiguous()
save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
json.dump({"architectures": [arch], "model_type": mtype, **common, **routing},
          open(out / "config.json", "w"), indent=1)

ids = [5, 17, 40, 12, 9, 51, 33, 20, 61, 7, 28, 44]
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), dim=-1)
json.dump({"ids": ids,
           "token_logprobs": [round(lp[i, ids[i + 1]].item(), 7) for i in range(len(ids) - 1)]},
          open(out / "reference.json", "w"))
print(out, sum(t.numel() * 2 for t in sd.values()), "bytes of weights")
