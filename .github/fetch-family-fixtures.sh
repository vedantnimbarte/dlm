#!/usr/bin/env sh
# Fetch one real config.json + tokenizer.json per model family for
# `tests/family_fixtures.rs`. Weights are deliberately NOT fetched — the defects
# these guard against were all in parsing, so ~85 MB of metadata covers them.
# (It grew from ~35 MB when the five families the README claimed but nothing
# tested were added; the large entries are 128k-vocabulary tokenizer.json files.)
#
# Every source is **ungated**: no HF token, so this also runs on fork PRs. Where
# the canonical repo is gated (Gemma, Mistral, Llama-2), a public mirror of the
# same tokenizer is used instead; the tokenizer.json is byte-identical to the
# official one, which is the file under test.
set -eu

dest="${1:-models/fixtures}"

# family|repo|extra files beyond config.json + tokenizer.json
fetch() {
  name="$1"; repo="$2"
  dir="$dest/$name"
  mkdir -p "$dir"
  for f in config.json tokenizer.json tokenizer_config.json; do
    url="https://huggingface.co/$repo/resolve/main/$f"
    # tokenizer_config.json carries add_bos_token and is required for Gemma and
    # Llama; a family without one is not an error.
    if ! curl -fsSL --retry 5 --retry-all-errors "$url" -o "$dir/$f"; then
      rm -f "$dir/$f"
      [ "$f" = tokenizer_config.json ] || {
        echo "error: $repo is missing $f" >&2
        exit 1
      }
    fi
  done
  echo "  $name <- $repo ($(du -sh "$dir" | cut -f1))"
}

echo "fetching family fixtures into $dest"
# SentencePiece-style BPE, one per signalling convention:
fetch gemma-2     unsloth/gemma-2-2b-it              # Replace normalizer, no prepend
fetch llama-2     NousResearch/Llama-2-7b-chat-hf    # Sequence[Prepend, Replace]
fetch mistral     mistral-community/Mistral-7B-v0.2  # Metaspace pre-tokenizer
# Byte-level BPE, plus the MoE/MLA config shapes:
fetch deepseek-v2 deepseek-ai/DeepSeek-V2-Lite-Chat
fetch qwen2.5     Qwen/Qwen2.5-0.5B-Instruct

# Families the README claimed but no fixture covered. Each row of the support
# table should have one behind it; these are the five that did not.
#
# llama-3 matters most: "Llama 2 / 3 / 3.1 / 3.2" is one table row, but Llama 3
# dropped SentencePiece for a 128k byte-level vocabulary and added `llama3` RoPE
# scaling. The llama-2 fixture covers neither, and a rope_scaling block that
# parses wrong yields fluent nonsense rather than an error.
fetch llama-3     unsloth/Llama-3.2-1B-Instruct               # llama3 rope scaling
fetch gemma-1     unsloth/gemma-1.1-2b-it                     # (1+w) norm, no softcap
fetch qwen3       Qwen/Qwen3-0.6B                             # explicit head_dim != derived
fetch qwen2-moe   Qwen/Qwen1.5-MoE-A2.7B-Chat                 # gated shared expert
# Mixtral's own repos are gated; this is a Mixtral-layout checkpoint, which is
# what the naming branch under test actually keys on.
fetch mixtral     NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO # block_sparse_moe
# Fused projections: qkv_proj and gate_up_proj. The weight-splitting itself is
# covered by tests/phi3_fused.rs, which needs no checkpoint; this is the config
# half.
fetch phi-3       microsoft/Phi-3-mini-4k-instruct            # fused qkv/gate_up
echo "done: $(du -sh "$dest" | cut -f1) total"
