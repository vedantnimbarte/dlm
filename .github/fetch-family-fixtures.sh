#!/usr/bin/env sh
# Fetch one real config.json + tokenizer.json per model family for
# `tests/family_fixtures.rs`. Weights are deliberately NOT fetched — the defects
# these guard against were all in parsing, so ~35 MB of metadata covers them.
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
echo "done: $(du -sh "$dest" | cut -f1) total"
