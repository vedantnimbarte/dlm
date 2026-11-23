"""The qwen35_moe fixture as llama.cpp's converter writes a Qwen3.5/3.6-MoE GGUF
(`qwen35moe`), for `tests/qwen35_parity.rs`: the same weights, transformed the
way `convert_hf_to_gguf.py` (conversion/qwen.py: Qwen3NextModel,
_LinearAttentionVReorderBase, _QwenMtpMixin) transforms them:

* value heads of every linear-attention tensor reordered from grouped by key
  head to tiled (in_proj_qkv's v rows, in_proj_z, in_proj_a/b, A, dt_bias, the
  conv's v channels, out_proj's columns);
* `ssm_a` = -exp(A_log); `dt_bias` as `ssm_dt.bias`; the conv squeezed to [C, K];
* `+1` added to every `(1 + w)` norm (not the gated `ssm_norm`), the MTP ones too;
* the stacked `gate_up_proj` split into `ffn_gate_exps` / `ffn_up_exps`;
* the MTP head after the model's layers as `blk.{n}`, its own tensors `nextn.*`.

usage (needs only numpy and gguf-py):
  python tests/fixtures/make_qwen35_gguf.py tests/fixtures/qwen35_moe
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import gguf

d = Path(sys.argv[1])
raw = (d / "model.safetensors").read_bytes()
n = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8 + n])
data = raw[8 + n:]
DT = {"F16": np.float16, "F32": np.float32}
T = {}
for k, v in header.items():
    if k == "__metadata__":
        continue
    a, b = v["data_offsets"]
    T[k] = np.frombuffer(data[a:b], dtype=DT[v["dtype"]]).reshape(v["shape"]).astype(np.float32)

cfg = json.load(open(d / "config.json"))["text_config"]
L = cfg["num_hidden_layers"]
nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
r = nv // nk


def tiled(t, dim, hd):
    """_reorder_v_heads: [nk, r, hd] -> [r, nk, hd] along `dim`."""
    t = np.moveaxis(t, dim, 0)
    s = t.shape
    t = t.reshape(nk, r, hd, *s[1:]).swapaxes(0, 1).reshape(s)
    return np.moveaxis(t, 0, dim)


out = {}
def put(name, t, f16=True):
    out[name] = t.astype(np.float16 if f16 and t.ndim >= 2 else np.float32)

def layer(src, blk):
    g = lambda s: T[f"{src}.{s}"]
    put(f"{blk}.attn_norm.weight", g("input_layernorm.weight") + 1)
    put(f"{blk}.post_attention_norm.weight", g("post_attention_layernorm.weight") + 1)
    if f"{src}.linear_attn.A_log" in T:
        la = lambda s: g(f"linear_attn.{s}")
        qkv = la("in_proj_qkv.weight")
        kd = nk * dk
        put(f"{blk}.attn_qkv.weight", np.concatenate([qkv[:2 * kd], tiled(qkv[2 * kd:], 0, dv)]))
        put(f"{blk}.attn_gate.weight", tiled(la("in_proj_z.weight"), 0, dv))
        put(f"{blk}.ssm_beta.weight", tiled(la("in_proj_b.weight"), 0, 1))
        put(f"{blk}.ssm_alpha.weight", tiled(la("in_proj_a.weight"), 0, 1))
        put(f"{blk}.ssm_a", tiled(-np.exp(la("A_log"))[:, None], 0, 1)[:, 0])
        put(f"{blk}.ssm_dt.bias", tiled(la("dt_bias")[:, None], 0, 1)[:, 0])
        conv = la("conv1d.weight")[:, 0, :]
        put(f"{blk}.ssm_conv1d.weight", np.concatenate([conv[:2 * kd], tiled(conv[2 * kd:], 0, dv)]), f16=False)
        put(f"{blk}.ssm_norm.weight", la("norm.weight"))
        put(f"{blk}.ssm_out.weight", tiled(la("out_proj.weight"), 1, dv))
    else:
        sa = lambda s: g(f"self_attn.{s}")
        put(f"{blk}.attn_q.weight", sa("q_proj.weight"))
        put(f"{blk}.attn_k.weight", sa("k_proj.weight"))
        put(f"{blk}.attn_v.weight", sa("v_proj.weight"))
        put(f"{blk}.attn_output.weight", sa("o_proj.weight"))
        put(f"{blk}.attn_q_norm.weight", sa("q_norm.weight") + 1)
        put(f"{blk}.attn_k_norm.weight", sa("k_norm.weight") + 1)
    put(f"{blk}.ffn_gate_inp.weight", g("mlp.gate.weight"), f16=False)
    put(f"{blk}.ffn_gate_inp_shexp.weight", g("mlp.shared_expert_gate.weight")[0])
    for p in ["gate", "up", "down"]:
        put(f"{blk}.ffn_{p}_shexp.weight", g(f"mlp.shared_expert.{p}_proj.weight"))
    gu = g("mlp.experts.gate_up_proj")
    inter = gu.shape[1] // 2
    put(f"{blk}.ffn_gate_exps.weight", gu[:, :inter])
    put(f"{blk}.ffn_up_exps.weight", gu[:, inter:])
    put(f"{blk}.ffn_down_exps.weight", g("mlp.experts.down_proj"))

put("token_embd.weight", T["model.language_model.embed_tokens.weight"])
put("output_norm.weight", T["model.language_model.norm.weight"] + 1)
put("output.weight", T["lm_head.weight"])
for i in range(L):
    layer(f"model.language_model.layers.{i}", f"blk.{i}")
layer("mtp.layers.0", f"blk.{L}")
put(f"blk.{L}.nextn.eh_proj.weight", T["mtp.fc.weight"])
put(f"blk.{L}.nextn.enorm.weight", T["mtp.pre_fc_norm_embedding.weight"] + 1)
put(f"blk.{L}.nextn.hnorm.weight", T["mtp.pre_fc_norm_hidden.weight"] + 1)
put(f"blk.{L}.nextn.shared_head_norm.weight", T["mtp.norm.weight"] + 1)

w = gguf.GGUFWriter(str(d / "qwen35_moe.gguf"), "qwen35moe")
a = "qwen35moe"
H = cfg["hidden_size"]
rope = cfg["rope_parameters"]
w.add_uint32(f"{a}.block_count", L + 1)
w.add_uint32(f"{a}.context_length", cfg["max_position_embeddings"])
w.add_uint32(f"{a}.embedding_length", H)
w.add_uint32(f"{a}.attention.head_count", cfg["num_attention_heads"])
w.add_uint32(f"{a}.attention.head_count_kv", cfg["num_key_value_heads"])
w.add_uint32(f"{a}.attention.key_length", cfg["head_dim"])
w.add_uint32(f"{a}.attention.value_length", cfg["head_dim"])
w.add_float32(f"{a}.rope.freq_base", rope["rope_theta"])
w.add_uint32(f"{a}.rope.dimension_count", int(cfg["head_dim"] * rope["partial_rotary_factor"]))
w.add_array(f"{a}.rope.dimension_sections", rope["mrope_section"] + [0])
w.add_float32(f"{a}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
w.add_uint32(f"{a}.expert_count", cfg["num_experts"])
w.add_uint32(f"{a}.expert_used_count", cfg["num_experts_per_tok"])
w.add_uint32(f"{a}.expert_feed_forward_length", cfg["moe_intermediate_size"])
w.add_uint32(f"{a}.expert_shared_feed_forward_length", cfg["shared_expert_intermediate_size"])
w.add_uint32(f"{a}.ssm.conv_kernel", cfg["linear_conv_kernel_dim"])
w.add_uint32(f"{a}.ssm.state_size", dk)
w.add_uint32(f"{a}.ssm.group_count", nk)
w.add_uint32(f"{a}.ssm.time_step_rank", nv)
w.add_uint32(f"{a}.ssm.inner_size", nv * dv)
w.add_uint32(f"{a}.full_attention_interval", cfg["full_attention_interval"])
w.add_uint32(f"{a}.nextn_predict_layers", 1)
w.add_string("tokenizer.ggml.model", "gpt2")
w.add_string("tokenizer.ggml.pre", "qwen35")
toks = [f"t{i}" for i in range(cfg["vocab_size"])]
w.add_array("tokenizer.ggml.tokens", toks)
w.add_array("tokenizer.ggml.token_type", [1] * len(toks))
w.add_array("tokenizer.ggml.merges", ["t 0"])
w.add_uint32("tokenizer.ggml.eos_token_id", 1)
w.add_uint32("tokenizer.ggml.bos_token_id", 0)
w.add_uint32("general.file_type", 1)
for name, t in out.items():
    w.add_tensor(name, np.ascontiguousarray(t))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(d / "qwen35_moe.gguf", sum(t.nbytes for t in out.values()), "bytes")
