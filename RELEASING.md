# Releasing

Push a `v*` tag and `.github/workflows/release.yml` builds the prebuilt binaries
(CPU + static-CUDA) and attaches them to the GitHub Release. `install.sh` /
`install.ps1` download those, so **whatever is tagged is what every new user
gets** — the one-liner does not build from source.

## What CI already proves

Every push runs: the CPU suite on Linux and Windows, `real_model` (a real
checkpoint must answer correctly), `gptq_model` (a real GPTQ export must decode
to weights that still mean something), clippy, and a `cargo check` of the CUDA
FFI.

## What CI cannot prove — do these by hand

**CI has no GPU.** It type-checks the CUDA FFI and never executes a single
device kernel. Everything below is therefore manual, and a green CI run is *not*
evidence that the `-cuda-static` binary you are about to ship works at all.

On a machine with an NVIDIA GPU and the CUDA toolkit (nvcc):

```sh
# 1. GPU<->CPU parity, including the int4/int8 decoders and the streamed kernel.
#    A silent layout drift between host and device corrupts weights rather than
#    erroring, so this is the check that catches it.
cargo test --release --features cuda-kernels

# 2. End-to-end on a real model, on the GPU. Tests can pass while the served
#    engine emits nonsense; only reading the output catches that.
cargo run --release --features cuda-kernels -- \
  serve --model-path models/<a-real-model> --device gpu --port 8000
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Capital of France? One word."}],"max_tokens":8,"temperature":0}'
# expect: Paris

# 3. The product claim itself: a model BIGGER than the card, quantized to fit.
cargo run --release --features cuda-kernels -- \
  serve --model-path models/<a-3B-or-larger> --device gpu --quant int4 --port 8000
# expect: coherent output, and VRAM under the card's limit
```

All three must pass on real hardware before tagging.

**Vary the family, not just the size.** Every blocker found before 0.3.0 lived in
`config.json` or `tokenizer.json` — a duplicate key that made Gemma2 unparseable,
a SentencePiece vocabulary encoded with byte-level BPE rules, an expert-count
field spelled differently by DeepSeek. The in-code test fixtures never touch
either file, and CI's only real checkpoint is Qwen: dense, byte-level BPE, no
BOS. A green suite says nothing about a family whose checkpoint nobody loaded.
Run check 2 against one model per family you intend to claim support for.

As of 0.3.0 that meant, by hand: Qwen2.5-0.5B, Qwen3-0.6B, a Qwen2.5 GPTQ-Int4
export, gemma-2-2b-it, gemma-1.1-2b-it, Mistral-7B-Instruct-v0.1,
Qwen1.5-MoE-A2.7B (14.3 B) and DeepSeek-V2-Lite-Chat (15.7 B) — the last two also
covering check 3, a model far larger than the card at `--quant int4`.

## Known gaps — read before you tag

These are not TODOs; they are the honest limits of what has been verified. A
release is a claim, and these bound it.

- **Only Turing-class hardware has run this code**: a GTX 1650 (4 GB). Ampere,
  Ada, Blackwell — anything with a different warp/SM profile — are unexercised.
- **Multi-GPU and distributed serving** are untested on real hardware.
- **Any context near the 8192 default is untested.** This is the sharpest one
  left: Gemma2 alternates windowed and global attention layers, and a window is
  only *reachable* past `sliding_window` tokens (4096 on gemma-2). A bug there is
  invisible in every short prompt — which is exactly how one shipped before, the
  device path clipping layers that must see full history. Decode past 4096 tokens
  on a Gemma2 model before trusting long-context output.
- **DeepSeek-V2/V3 (MLA + MoE) has only been run on the CPU path.** The README
  advertises it on the streaming GPU path; that has never been executed against
  real weights, because a 4 GB card cannot meaningfully stream a 30 GB
  checkpoint. The synthetic GPU parity tests pass, which is not the same claim.
- **Falcon has never been run.** The loader, config handling and refusals are in
  place and covered by a real `falcon-7b` config fixture, but no Falcon
  checkpoint has been executed: the smallest variant with `alibi: false` is
  14 GB, and the 1B `falcon-rw-*` uses ALiBi, which dlm refuses. So Falcon is
  *config-verified*, not *output-verified* — a weaker claim than Phi-3 (whose
  weight mapping is proven by a same-weights-twice equivalence test) or GPT-2
  (which answers correctly on the real checkpoint). Run check 2 against a Falcon
  before claiming it works.
- **Llama 2 is untested.** It shares Mistral's SentencePiece tokenizer shape,
  which is verified, so it is expected to work — expected, not demonstrated.

## Tag

```sh
# 1. Bump `version` in Cargo.toml, refresh Cargo.lock (`cargo build`), commit.
# 2. Run the manual GPU checks above.
# 3. Then:
git tag vX.Y.Z && git push origin vX.Y.Z
```

Watch the release run: a failed `build-cuda-static` leg leaves the CPU assets
published and the GPU ones missing, and `install.sh` will then hand NVIDIA users
a CPU build without saying why.
