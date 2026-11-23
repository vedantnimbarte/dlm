"""A tiny random Gemma 4 checkpoint (f16) and transformers' log-probabilities
for it, as a test fixture: dlm must score the same token sequence the same.

usage: python tests/fixtures/make_gemma4.py tests/fixtures/gemma4

The shape is E2B/E4B's in miniature: windowed and global layers with different
head widths and RoPE (the global ones proportional, rotating a quarter of each
head), a shared-KV tail reading one layer of each kind, E2B's double-wide MLP on
that tail, per-layer inputs, layer scalars, and a final-logit softcap small
enough to bend the logits. The sequence is three times the window. The
checkpoint is laid out as Google's multimodal one is: the language model under
`model.language_model.*`, the unused K/V of the shared layers still present,
and stand-in vision and audio tensors that must be skipped.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(4)
text = dict(
    model_type="gemma4_text", vocab_size=64, vocab_size_per_layer_input=64,
    hidden_size=32, intermediate_size=48, num_hidden_layers=6,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
    sliding_window=4, num_kv_shared_layers=2, use_double_wide_mlp=True,
    hidden_size_per_layer_input=8, hidden_activation="gelu_pytorch_tanh",
    rope_parameters={
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                           "rope_theta": 1000000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
    final_logit_softcapping=5.0, rms_norm_eps=1e-6, max_position_embeddings=64,
    tie_word_embeddings=True, attention_k_eq_v=False, enable_moe_block=False,
    bos_token_id=2, eos_token_id=1, pad_token_id=0, dtype="float16",
)
model = Gemma4ForCausalLM(
    Gemma4TextConfig(**{k: v for k, v in text.items() if k != "model_type"})).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.25)
        else:
            # Gemma 4's norms scale by `w` itself, not `1 + w`.
            p.normal_(1.0, 0.2)
        # Round to what the f16 checkpoint holds, so the reference is computed
        # from exactly the weights dlm reads.
        p.copy_(p.half().float())
    for name, b in model.named_buffers():
        if name.endswith("layer_scalar"):
            b.uniform_(0.5, 1.5)
            b.copy_(b.half().float())

sd = {}
for name, t in model.state_dict().items():
    if name.startswith("model.") and "rotary_emb" not in name and "embed_scale" not in name:
        sd["model.language_model." + name[len("model."):]] = t.half().contiguous()
first_shared = text["num_hidden_layers"] - text["num_kv_shared_layers"]
g = torch.Generator().manual_seed(5)
for layer in range(first_shared, text["num_hidden_layers"]):
    # Google's checkpoint still carries K/V for the layers that read another
    # layer's cache; transformers ignores them, and so must dlm.
    kv = 32 if (layer + 1) % 3 == 0 else 16
    for proj in ("k_proj", "v_proj"):
        sd[f"model.language_model.layers.{layer}.self_attn.{proj}.weight"] = torch.randn(
            kv, 32, generator=g).half()
    sd[f"model.language_model.layers.{layer}.self_attn.k_norm.weight"] = torch.randn(
        kv, generator=g).half()
# Stand-ins for the towers dlm does not run. Their `layers.N` names must not be
# taken for decoder layers.
sd["model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"] = torch.randn(4, 4).half()
sd["model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight"] = torch.randn(4, 4).half()
sd["model.embed_vision.embedding_projection.weight"] = torch.randn(4, 4).half()
sd["model.embed_audio.embedding_projection.weight"] = torch.randn(4, 4).half()
save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
json.dump({"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
           "text_config": text, "vision_config": None, "audio_config": None,
           "eos_token_id": [1, 106], "tie_word_embeddings": True},
          open(out / "config.json", "w"), indent=1)

ids = [2, 17, 40, 12, 9, 51, 33, 20, 61, 7, 28, 44]
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
json.dump({"ids": ids,
           "token_logprobs": [round(lp[i, ids[i + 1]].item(), 7) for i in range(len(ids) - 1)],
           "next_logprobs": [round(v, 7) for v in lp[-1].tolist()]},
          open(out / "reference.json", "w"))
print(out, sum(t.numel() * 2 for t in sd.values()), "bytes of weights")
