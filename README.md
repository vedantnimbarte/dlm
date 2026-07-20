<p align="center">
  <img src="logo.png" alt="dlm logo" width="200">
</p>

# dlm

<p align="center">
  <strong><a href="https://vedantnimbarte.github.io/dlm-frontend/">Docs</a></strong> ·
  <a href="https://vedantnimbarte.github.io/dlm-frontend/get-started/">Get started</a> ·
  <a href="https://vedantnimbarte.github.io/dlm-frontend/calculator/">Can I run it?</a> ·
  <a href="https://vedantnimbarte.github.io/dlm-frontend/docs/quantization/">Quantization</a>
</p>

**dlm (Dynamic LLM)** — a dynamic layer-streaming inference engine. **Run models
bigger than your VRAM**, whatever card you have — 4 GB, 6 GB, 8 GB, 16 GB, or
more. dlm streams transformer layers in and out of VRAM instead of loading the
whole model at once, so your GPU can run LLMs several times larger than it could
normally hold.

Rather than keeping every weight in VRAM, `dlm` keeps only a small window of
transformer blocks resident and streams the rest in over the PCIe bus as the GPU
works through them. The model that wouldn't load at all now runs — but streaming
is **slow**, and honestly so: moving most of a model across the bus every token
costs far more than the arithmetic does. The fewer layers fit, the more it
streams and the slower it gets.

So reach for [`--quant`](#weight-precision---quant) first. Quantizing the weights
at load shrinks each layer 2–4x, which usually means **more of the model stays
resident and streaming shrinks or stops entirely** — worth far more than any
amount of making streaming itself faster. On a 4 GB card, a 3B model (which does
not fit in 16-bit) goes from 0.024 tok/s streamed to **4.2 tok/s** fully resident
at `--quant int4`. Streaming is the fallback for what still doesn't fit.

It's not a 16 GB tool. It's for **anyone the VRAM wall has been telling "no"** —
the 4 GB laptop GPU, the 6 GB gaming card, the 8 GB workstation — giving each of
them the headroom to run models above their weight class. Tune the resident
window to your card with `--vram-budget-gb` and `--safety-margin-gb` (drop the
safety cushion on a small card to fit more layers).

---

## Install

One line — downloads a prebuilt binary for your platform and installs it. No
clone, no build, no Rust toolchain. Every download is checked against a published
`.sha256` before it is unpacked or run, so a corrupted or truncated download is
caught. Note this is an **integrity** check, not authenticity: the checksum is
served from the same GitHub release as the binary, so it guards against corruption
in transit, not a compromised release. Signed releases are on the roadmap.

**Linux / macOS** (installs to `~/.local/bin`):

```sh
curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/dlm/main/install.sh | sh
```

**Windows** (installs to `%LOCALAPPDATA%\Programs\dlm` and adds it to PATH):

```powershell
irm https://raw.githubusercontent.com/vedantnimbarte/dlm/main/install.ps1 | iex
```

Prebuilt targets: Linux x86-64/arm64, macOS Apple Silicon, Windows x86-64. (Intel
Macs: build from source with `cargo install`.)

On **x86-64 Linux or Windows with an NVIDIA GPU**, the installer picks the
**GPU (CUDA) build** automatically and runs on the GPU by default — pass
`--device cpu` to `serve` or `generate` to force CPU per-command. Everywhere else
(no NVIDIA GPU, arm64, macOS) it installs the portable **CPU build**.

The GPU build **statically embeds the CUDA runtime**, so it needs only the
**NVIDIA driver** — no CUDA toolkit install. An **AMD GPU** gets the CPU build for
now (AMD GPU support is planned; see [Building for GPU](#building-for-gpu-nvidia--amd)).

- Force the CPU build: `DLM_CPU=1 curl … | sh` (or set `DLM_CPU=1` before the
  Windows one-liner).
- If the GPU build won't start (driver missing or too old), the installer says so
  and falls back to the CPU build on its own.

Then:

```sh
dlm search llama-3.2   # find models on the Hugging Face hub
dlm pull <org/model>   # download one locally (no hf CLI needed)
dlm doctor             # check your machine + run a self-test
dlm --help             # search, pull, serve, generate, profile, tokenize, doctor, completions
```

Set `DLM_INSTALL_DIR` to change the location. To build the GPU binary yourself,
see [Building for GPU](#building-for-gpu-nvidia--amd).

Rust users can also `cargo install --git https://github.com/vedantnimbarte/dlm`
(builds from source).

Shell completions (bash/zsh/fish/elvish/powershell) — `dlm completions <shell>`
prints a script to stdout. E.g. for bash:

```sh
dlm completions bash > ~/.local/share/bash-completion/completions/dlm
```

To update to the latest release (just reinstalls; same env as install):

```sh
curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/dlm/main/update.sh | sh
```

To uninstall (removes the binary; respects `DLM_INSTALL_DIR`):

```sh
curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/dlm/main/uninstall.sh | sh
```

On **Windows** (removes the binary and its user-PATH entry):

```powershell
irm https://raw.githubusercontent.com/vedantnimbarte/dlm/main/uninstall.ps1 | iex
```

---

## Table of contents

- [Install](#install)
- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Build & run locally](#build--run-locally)
- [Running the tests](#running-the-tests)
- [The VRAM budget math](#the-vram-budget-math)
- [Building for GPU (NVIDIA / AMD)](#building-for-gpu-nvidia--amd)
- [Weight precision (`--quant`)](#weight-precision---quant)
- [Distributed & scaling](#distributed--scaling)
- [Contributing & license](#contributing--license)

---

## How it works

`dlm` partitions VRAM into three regions:

```
┌────────────────────────────────────────────────────────┐
│                      GPU VRAM                          │
├───────────────────┬───────────────────┬────────────────┤
│   PINNED ZONE     │   STREAMING ZONE  │  CACHE ZONE    │
│  • Embedding Block│  • Resident window│ • Paged KV     │
│  • LM Head / Norm │    of N layers    │ • Intermediate │
│  • Draft Model    │    (LRU, async)   │   Residuals    │
└───────────────────┴───────────────────┴────────────────┘
```

- **Pinned Zone** — embedding, LM head, and norms stay resident permanently
  (moving them each step would thrash the PCIe bus).
- **Streaming Zone** — an LRU of `N` layers (`--resident-layers`, else the VRAM
  plan's budget). A miss uploads that layer on a dedicated copy stream, so the
  transfer overlaps compute on the default stream, and evicts the least-recently
  used.
- **Cache Zone** — PagedAttention KV cache + residual activations. KV for *all*
  layers stays resident even as weights stream, so attention always sees full
  history.

The data path for a streamed layer:

```
mmap weights ──► host RAM cache ──► pinned staging buffer ──► streaming-zone buffer ──► compute
   (NVMe)        (--ram-cache-gb)     (page-locked host)          (VRAM)
```

Memory-mapping skips the OS read-buffer copy; the optional host-RAM cache
(`--ram-cache-gb`, off by default) keeps materialized layers across token steps
so a miss doesn't re-read and re-decode them; and the page-locked (pinned) host
buffer lets the PCIe controller DMA into VRAM on the copy stream.

Be clear-eyed about the limit: with a window smaller than the model, layers are
touched in order and evicted long before they come round again, so the window
does not really *cache* — its job is to overlap the next upload with the current
compute. What streaming costs is bandwidth, and the only way to pay less is to
send fewer bytes ([`--quant`](#weight-precision---quant)).

The transformer math sits behind a block-level `ComputeKernel` trait (`run_block`
runs one decoder block for one token), and a `ForwardOrchestrator` drives a
sequence through the model autoregressively: per token it reserves KV budget,
then calls the kernel for each layer, threading each layer's real K/V history.

Three interchangeable kernels sit behind the trait:

- **`CpuKernel`** — the real math: a Llama-style decode block (RMSNorm, RoPE,
  grouped-query attention over the KV history, SwiGLU MLP) in
  [`src/forward/cpu.rs`](src/forward/cpu.rs). Plugged into the orchestrator it
  gives a fully-connected — if slow, single-token — **CPU forward path**, and
  serves as the correctness oracle and porting spec for the GPU kernel.
- **`StubKernel`** — a trivial deterministic kernel for testing the
  orchestration (KV growth, per-layer iteration) in isolation.
- **`GpuKernel`** — a CUDA `run_block` on the device (feature `cuda-kernels`).
  The transformer math is in [`src/gpu/kernels.cu`](src/gpu/kernels.cu) (RMSNorm,
  RoPE, GQA attention, SwiGLU — mirroring the CPU oracle op-for-op); the Rust
  side ([`src/forward/gpu.rs`](src/forward/gpu.rs)) uploads weights to VRAM and
  launches the block. **KV history stays resident in VRAM** — the new token's
  K/V is appended in place and attention reads it directly, so only the hidden
  vector crosses the PCIe bus per token, not the whole history. Requires nvcc + a
  GPU; validated against the CPU kernel.

Around the stack, [`src/generate.rs`](src/generate.rs) closes the loop:
`token → embedding → transformer stack → final RMSNorm → LM head → logits →
sample → next token`. With the `CpuKernel` this is a complete, end-to-end (if
slow, single-sequence) **CPU inference path** — prompt tokens are prefilled into
the KV history, then new tokens are generated greedily until an EOS or the token
limit.

The GPU-specific paths are behind `cuda` / `rocm`
[feature flags](#building-for-gpu-nvidia--amd); with neither, the engine uses a
page-aligned host fallback with the same layout contract, so the logic runs on
any machine.

## Prerequisites

- **Rust** 1.75+ (2021 edition). Install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Linux/macOS
  ```
  On Windows, download and run [`rustup-init.exe`](https://rustup.rs/).
- A C toolchain for building native dependencies (usually already present):
  - Linux: `build-essential`
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Windows: the Visual Studio C++ Build Tools (MSVC)
- **(Optional, for the GPU path)** NVIDIA CUDA Toolkit 12.x with `cudart` on the
  library path. (AMD ROCm is memory-only for now — see
  [Building for GPU](#building-for-gpu-nvidia--amd) — so it needs no toolkit to
  build.)

  Not required for building, testing, or the demo — the host fallback needs no GPU.

## Build & run locally

Clone and build (host-only, no GPU needed):

```bash
git clone <your-fork-url> dlm
cd dlm
cargo build            # debug build
cargo build --release  # optimized build
```

The binary exposes these subcommands:

```bash
cargo run -- --help          # top-level help
cargo run -- search llama    # search the Hugging Face hub for models
cargo run -- pull <org/model># download a model locally (via curl, no hf CLI)
cargo run -- profile         # profile a sample 70B-class model (no GPU needed)
cargo run -- serve --help    # full serve flag list (specs §4)
cargo run -- generate --help # end-to-end CPU generation on a synthetic model
cargo run -- tokenize --help # byte-level BPE encode/decode round-trip
cargo run -- doctor          # check machine (GPU/VRAM) + run an inference self-test
```

**`search` / `pull`** — find and download models straight from the
[Hugging Face hub](https://huggingface.co), no `hf` CLI or manual file-grabbing
needed. `pull` shells out to `curl` (built into Linux, macOS, and Windows 10/11)
to fetch only the files dlm loads (`config.json`, `*.safetensors`, tokenizer):

```bash
cargo run -- search llama-3.2                 # most-downloaded matches, safetensors only
cargo run -- pull Qwen/Qwen2.5-0.5B-Instruct  # → ./models/Qwen2.5-0.5B-Instruct
cargo run -- serve --model-path ./models/Qwen2.5-0.5B-Instruct
```

A full HF URL works in place of the `org/model` id. Use `--local-dir` to change
where it lands, and `--token` (or `$HF_TOKEN`) for gated/private repos. Only
safetensors checkpoints load — GGUF/PyTorch-only repos are rejected with a clear
message.

**`profile`** — with no `--model-path` it profiles a representative
Llama-3-70B-class model against a simulated 16 GB card:

```bash
cargo run -- profile
```

Example output:

```
dlm v0.2.1
  gpu backend  : none (host fallback)
  host page    : 4096 bytes

model source : built-in Llama-3-70B-class sample
geometry     : 80 layers, hidden 8192, 64 q-heads / 8 kv-heads, head_dim 128
quantization : Fp16 (2 bytes/param), ~70.6 B params

free VRAM    : simulated 16 GiB (no GPU device)

── VRAM PLAN ─────────────────────────────────
  M_free           :    16384.0 MiB
  M_safety         :     1536.0 MiB
  M_kv_total       :     2560.0 MiB
  pinned_zone      :        0.0 MiB
  M_layer_weight   :     1682.1 MiB
  usable           :    12288.0 MiB
  ▶ layers_to_load :          7 / 80
  ▶ resident       :       8.8%
──────────────────────────────────────────────

kv cache     : 512 paged blocks × 16 tok, 5.00 MiB/block → 8192 token capacity
swap cycle   : 12 streaming pass(es), window of 7 layer(s)
pipeline     : 4 steps, 2 overlapped (DMA hidden under compute)
```

The sample assumes 16-bit weights, so a layer is 1.7 GiB and only 7 of 80 fit.
`--quant int4` quantizes the weights at load, dropping a layer to 420 MiB so the
same 16 GiB holds 29 (`int8`: 841 MiB, 14) — see
[weight precision](#weight-precision---quant).

Point it at a real model directory (containing `config.json` and
`*.safetensors` shards) to map the actual weights and profile from measured
layer sizes:

```bash
cargo run -- profile --model-path /path/to/models/Llama-3-70B-Instruct
```

**`serve`** — starts the HTTP API server for a model. It speaks **two dialects**,
so clients like Open WebUI, the OpenAI SDKs, and the Anthropic SDKs all talk to
`dlm` unchanged with nothing but a base-URL change:

| Method | Path | |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI chat completion (SSE with `"stream": true`) |
| `POST` | `/v1/completions` | OpenAI legacy completion on a raw `prompt` |
| `GET` | `/v1/models` | lists the served model |
| `POST` | `/v1/messages` | Anthropic Messages API (SSE with `"stream": true`) |
| `POST` | `/v1/messages/count_tokens` | token count for an Anthropic-shaped request, no generation |
| `GET` | `/metrics` | Prometheus counters (**needs the key** under `--api-key`) |
| `GET` | `/health` (also `/`, `/healthz`) | liveness — the only routes that stay open |

```bash
cargo run -- serve \
    --model-path /path/to/small-model \
    --context-length 8192 \
    --port 8000 \
    --host 127.0.0.1

# then, from another shell (add "stream": true for token-by-token SSE):
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dlm","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# the same model over the Anthropic dialect:
curl http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"dlm","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

On `/v1/messages`, a top-level `system` field is folded in as a leading system
message, and `content` may be a string or a list of text blocks. Streaming
follows the sequence Anthropic clients expect — `message_start`,
`content_block_start`, a run of `content_block_delta`, `content_block_stop`,
`message_delta`, `message_stop`.

Concurrent requests are continuously batched by a background scheduler, and
`"stream": true` streams the reply as Server-Sent Events.

With `--distributed-mode worker` the process instead serves its layer shard to a
master over TCP (see [Distributed & scaling](#distributed--scaling)).

**`generate`** — drives the full CPU generation loop (embedding → transformer
stack → LM head → greedy sampling) on a **randomly-initialized** synthetic model.
There is no checkpoint loader or tokenizer yet, so it operates on token ids and
the output is deterministic-but-meaningless — it exercises the whole pipeline
end-to-end through the binary:

```bash
cargo run -- generate --prompt 1,2,3 --max-new-tokens 8 --seed 42
# prompt       : [1, 2, 3]
# generated    : [19, 6, 29, 6, 29, 6, 29, 5]
```

Point `generate` at a **real** checkpoint to run it on CPU. The loader
([`src/loader.rs`](src/loader.rs)) reads the standard HuggingFace-named tensors
(F32/F16/BF16) out of the mapped safetensors and materializes the transformer,
embedding, and LM head:

```bash
cargo run -- generate --model-path /path/to/small-model --prompt 1,2,3
```

Provide a **text** prompt (tokenized with a byte-level BPE tokenizer) instead of
raw ids. It uses the model directory's `vocab.json` + `merges.txt` if present,
otherwise a raw byte tokenizer:

```bash
cargo run -- generate --model-path /path/to/small-model --text "Hello"
```

On a `cuda-kernels` build generation runs on the **GPU by default** (`GpuKernel`);
`--device cpu` forces the CPU kernel, and if no GPU is usable dlm warns and falls
back to CPU on its own. A CPU-only build defaults to CPU, and an explicit
`--device gpu` there errors with a clear message:

```bash
cargo run --features cuda-kernels -- generate --model-path /path/to/small-model              # GPU
cargo run --features cuda-kernels -- generate --model-path /path/to/small-model --device cpu # force CPU
```

The standalone `tokenize` subcommand shows the encoder round-trip:

```bash
cargo run -- tokenize --text "Hello, world!"
# ids        : [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
# round-trip : "Hello, world!" (ok)
```

### Supported models

dlm implements the **Llama-style decoder block** — dense and
**Mixture-of-Experts** — and reads the standard `model.layers.{i}.*` tensor
names. That covers:

| Family | Status |
|---|---|
| Llama 2 / 3 / 3.1 / 3.2 | supported (incl. `llama3` RoPE scaling, GQA, tied embeddings) |
| Mistral | supported (incl. sliding-window attention) |
| Qwen2 / Qwen2.5 | supported (incl. the Q/K/V attention biases) |
| Qwen3 (dense) | supported — per-head Q/K RMSNorm |
| Gemma (v1) | supported — `(1+w)` RMSNorm, GeGLU, `sqrt(hidden)` embedding scaling |
| Mixtral (and Mixtral-layout MoE) | supported — top-k routing over `block_sparse_moe.experts.*` |
| Qwen2-MoE / Qwen3-MoE | supported — routed experts + optional sigmoid-gated shared expert |
| GPT-2 / Falcon / other layouts | **not supported** — errors on unknown tensor names |
| DeepSeek-V2/V3 (MLA) | supported — Multi-head Latent Attention (compressed-latent KV, decoupled RoPE, YaRN). GPU covers MLA + a dense FFN; MLA + MoE and streamed-MLA on GPU still run on CPU |
| Gemma2 | **not supported** — needs attention logit softcapping + alternating sliding-window layers |

**MoE models** route each token through the top-k experts the router selects
(softmax over all experts, then top-k, then renormalized — the Mixtral/Qwen
recipe). **Both the GPU and the host (`--stream`) paths hold only a layer's
*core* resident — attention, router, and the shared expert — and pull the top-k
routed experts on demand**, cached per `(layer, expert)` and reused across
tokens. So a sparse model keeps memory near its *active* parameter count, not its
total: a Mixtral-8×7B layer inlines ~45 GB of experts, but only the two a token
uses are materialized. On GPU the expert cache is a VRAM budget
(`--expert-cache-gb`, defaulting to the VRAM left after the resident layer
window) so a fine-grained (128-expert) model can't overrun the card; on the host
it's a bounded RAM cache. Because expert choice is data-dependent it can't be
prefetched like the layer cycle, so a cold expert still costs a load; quantize
(`--quant`) to keep the hot set resident, and watch `dlm_stream_expert_*` in
`/metrics` for the hit rate. Expert weights honor `--quant` like dense weights.

Concurrent requests are continuously batched on every backend, including the GPU
paths — each in-flight sequence owns its own KV history, so batched decoding is
correct on the GPU as well as the CPU. KV lives in VRAM per sequence, so the KV
footprint grows with the batch size; size `--context-length` and the batch for
your card.

An unsupported architecture fails with a clear `UnknownTensor` error at load
rather than producing garbage. A config declaring a `rope_scaling` type dlm does
not implement is likewise **refused**, not silently ignored — the model was
*trained* with that scaling, so running without it yields fluent nonsense.

The loader handles **float checkpoints** (`.weight` in F32/F16/BF16). Attention
biases (`q_proj.bias`/`k_proj.bias`/`v_proj.bias`, which Qwen2 ships and Llama does
not) are loaded when present, and `rope_scaling` (`linear`, `llama3`) is applied
when the config declares it.

**Quantized checkpoints (GPTQ/AWQ) are refused, not silently mis-loaded.** The
4-bit dequantizer in [`src/quant/packed.rs`](src/quant/packed.rs) is round-trip
tested against dlm's own packer, but has never been validated against a real
export — and exporters disagree on the zero-point convention (AutoGPTQ stores
`zero - 1`) and on act-order column permutation. Getting either wrong yields
*plausible-looking but incorrect* weights, which is a far worse failure than an
honest error. Use an fp16/bf16 checkpoint. (Re-enabling this needs a real GPTQ
fixture plus a parity test — the code is still there behind the refusal.)

## Running the tests

```bash
cargo test              # unit + integration tests (host fallback)
cargo test -- --nocapture   # with println output
```

The suite covers the safetensors parser and zero-copy reads, the VRAM math
(against hand-computed values), pinned-buffer alignment, tensor classification,
the layer catalog, and the double-buffer schedule + execution correctness (that
the A/B swap never corrupts the window under compute).

## The VRAM budget math

The profiler decides how many transformer blocks fit resident at once:

```
                 ⌊ M_free − M_safety − M_kv_total − M_pinned ⌋
LayersToLoad  =  ─────────────────────────────────────────────
                              M_layer_weight
```

- **`M_free`** — free VRAM from the GPU runtime at runtime
  (`cudaMemGetInfo` / `hipMemGetInfo`; simulated off-GPU).
- **`M_safety`** — cushion for activation spikes (default **1.5 GiB**).
- **`M_kv_total`** — KV cache for the whole context, summed across **all** layers
  (their histories stay resident while weights stream):
  `2 × N_kv_heads × D_head × 2 bytes × L_context × N_layers`.
- **`M_pinned`** — permanent Pinned Zone cost (embedding + LM head + norms),
  measured from the real checkpoint when available.
- **`M_layer_weight`** — size of one streamed block: the largest measured block
  from the catalog, or a parameter-count estimate at bootstrap.

The result is clamped to `[1, N_layers]` — streaming needs at least one resident
slot, and never more than the model has.

## Building for GPU (NVIDIA / AMD)

The GPU path is vendor-neutral behind [`src/gpu`](src/gpu), selected by a Cargo
feature. Everything above the backend (storage, profiler, pipeline) is identical
across vendors.

> **GPU compute status:** The GPU stack is **backend-agnostic** — the same
> kernels (`src/gpu/kernels.cu`) and orchestration run on **NVIDIA (CUDA)** via
> `cuda-kernels` and **AMD (HIP)** via `rocm-kernels`; only the device runtime
> (malloc/copy/streams) differs. CUDA is verified on real hardware against the CPU
> oracle ([`tests/gpu_parity.rs`](tests/gpu_parity.rs)). The HIP path type-checks
> and compiles (`hipcc`), and needs the same parity run on an AMD card to be
> declared verified — validate with `dlm doctor` before trusting output.

| Feature | Vendor | Runtime | Env var |
|---|---|---|---|
| `cuda` | NVIDIA | `cudart` (dynamic) | `CUDA_PATH` |
| `cuda-kernels` | NVIDIA | dynamic `cudart` + compiled `kernels.cu` (nvcc) | `CUDA_PATH` |
| `cuda-static` | NVIDIA | **static** `cudart` baked in — runs on driver alone | `CUDA_PATH` |
| `rocm` | AMD | `amdhip64` — memory only, no compute yet (planned) | `ROCM_PATH` |
| _(none)_ | — | host fallback | — |

`cuda-kernels` additionally compiles [`src/gpu/kernels.cu`](src/gpu/kernels.cu)
with nvcc and enables `GpuKernel` (the device `run_block`). `cargo check
--features cuda-kernels` type-checks the Rust FFI without the toolkit; building a
binary that runs the kernels needs nvcc + a GPU.

**The prebuilt release ships the `cuda-static` build** (Linux and Windows
x86-64). Statically embedding the CUDA runtime means the toolkit is needed only
at *build* time — the shipped binary runs on any machine with just the NVIDIA
driver. `cuda-static` is a strict superset of the dynamic `cuda-kernels` build
(it also runs where the toolkit is present), so there is no separate dynamic
release asset; build the dynamic variant yourself with
`cargo install --git … --features cuda-kernels` if you specifically want it.

On a CUDA machine, `cargo test --features cuda-kernels` runs
[`tests/gpu_parity.rs`](tests/gpu_parity.rs), which decodes the same random model
on both `CpuKernel` and `GpuKernel` and asserts the hidden states match — the CPU
kernel is the correctness oracle for the device code.

Type-checking works without either toolkit installed; building/linking requires
the corresponding runtime on the link path:

```bash
cargo check --features cuda            # validates the CUDA FFI, no linking
cargo check --features rocm            # validates the ROCm/HIP FFI, no linking

CUDA_PATH=/usr/local/cuda cargo build --features cuda   # NVIDIA
ROCM_PATH=/opt/rocm        cargo build --features rocm   # AMD
```

The two GPU features are mutually exclusive. With one enabled, `PinnedBuffer`
allocates genuine page-locked memory (`cudaHostAlloc` / `hipHostMalloc`) and
`mem_get_info` reports the live device's free VRAM. With neither, buffers are
page-aligned host allocations (same layout contract, promotable in place later
via `cudaHostRegister` / `hipHostRegister`), so nothing about the pipeline shape
changes between builds.

### Running on unofficially-supported AMD cards

> **Not functional yet.** This applies only once AMD GPU compute lands (see the
> status note above). It is kept here as reference for that work — today the
> `rocm` build has no compute kernels and runs inference on the CPU.

ROCm ships kernels only for a short list of "officially supported" GPUs, but many
Radeon cards that aren't on that list share an ISA with one that is. The
`HSA_OVERRIDE_GFX_VERSION` env var tells ROCm to treat your GPU as the nearest
supported `gfx` target — no rebuild, just a runtime override on the `rocm` build:

```bash
# Set the gfx version of the nearest supported card in your GPU's family:
HSA_OVERRIDE_GFX_VERSION=10.3.0 dlm serve <model>
```

Pick the value from your GPU's architecture:

| Architecture | Example cards | `HSA_OVERRIDE_GFX_VERSION` |
|---|---|---|
| RDNA 3 | RX 7600 / 7700 XT | `11.0.0` |
| RDNA 2 | RX 6600 / 6700 / 6800 | `10.3.0` |
| RDNA 1 | RX 5500 / 5700 | `10.1.0` |
| Vega / GCN 5 | Vega 56/64, Radeon VII | `9.0.0` |

This is unofficial: it only works when the override matches your card's real ISA,
and a mismatch can hang or return garbage rather than error cleanly — verify with
`dlm doctor` (runs a self-test) before trusting output. It does **not** enable
most integrated APUs, whose iGPUs ROCm can't drive regardless. When in doubt,
`rocminfo` prints your GPU's actual `gfx` name.

## Weight precision (`--quant`)

**Omit it and dlm computes in the checkpoint's own precision** — it reads the
dtype the file actually holds (`bf16`/`f16` → 16-bit, `f32` → 32-bit) and
converts to f32 only in the register that consumes each weight. There is no
default to get wrong, and nothing is upsized: an f32 copy of bf16 weights is
lossless (every bf16 value is exactly an f32), so it would buy no precision while
doubling VRAM, PCIe per streamed layer, and GEMV bandwidth.

| `--quant` | bytes/param | what happens |
| --- | --- | --- |
| *(omitted)* | 2.0 / 4.0 | the checkpoint's own dtype — no conversion |
| `int4` | 0.5 | quantized at load — 4x smaller, coarsest (16 levels per group) |
| `int8` | 1.0 | quantized at load — 2x smaller, 256 levels per group |
| `fp16` / `f32` | 2.0 / 4.0 | accepted only if it *matches* the checkpoint |

Both quantized modes are group-affine in groups of 128:
`w = (code − zero[g]) × scale[g]`, decoded in-register by the same kernel that
reads the float dtypes. A scheme dlm cannot actually deliver is an error, not a
silent no-op — `--quant` never describes weights that don't exist.

**Quantizing is the lever that matters on a small card.** A smaller layer means
more of the model stays resident and each streamed layer costs less to move —
often it removes streaming entirely, which is worth more than making streaming
faster. On a 4 GB GTX 1650, Llama-3.2-3B (5.6 GiB of bf16 layers, i.e. a model
that does **not** fit):

| | layer | resident | tok/s |
| --- | --- | --- | --- |
| bf16, `--stream` | 192 MiB | 5 / 28 | 0.024 |
| `--quant int8` | 96 MiB | all 28 (no streaming) | **3.0** |
| `--quant int4` | 48 MiB | all 28 (no streaming) | **4.2** |

int4 is faster than int8 for the same reason it is smaller: the GEMV is
bandwidth-bound, so halving the bytes read halves the work. Pick int8 when int4
costs too much accuracy — its 256 levels per group track a weight's range ~17x
finer than int4's 16 — and pick int4 when you need the model to fit at all.

Both are lossy: group-affine rounding without the calibration GPTQ/AWQ use. So
check your model's output rather than assuming. Three caveats worth knowing:

- **Quantizing costs load time** (it runs over every weight) and reads the full
  16-bit tensors from disk regardless; the win is in VRAM and on the bus, not in
  what is read.
- **`--stream` + a quantized `--quant` currently re-quantizes a layer on every
  window miss**, which is far slower than either flag alone. Pair it with
  `--ram-cache-gb` (which caches the quantized layer), or drop `--stream` — once
  quantized, the model often no longer needs it.
- **A quantized weight is still lossy even if it fits.** Verify on your own
  prompts; a model that fits but answers worse is not obviously a win.

### Already-quantized checkpoints (GPTQ)

`--quant` quantizes a *float* checkpoint at load. A checkpoint that is **already**
4-bit is a different thing, and dlm loads those directly:

```sh
dlm pull Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 --local-dir models/qwen-gptq
dlm serve --model-path models/qwen-gptq --chat-template chatml
```

Its codes are *relabeled* into dlm's layout, not dequantized and re-quantized, so
the accuracy the export's calibration bought survives — a GPTQ int4 model is
generally better than the same model through `--quant int4`, which rounds to
nearest with no calibration. No `--quant` is needed (or accepted): the file's
precision is already int4.

Fully validated: **4-bit GPTQ with `desc_act: false`** and the classic `gptq`
checkpoint format — validated end-to-end against a real export
([`tests/gptq_model.rs`](tests/gptq_model.rs)), and relabeled straight into dlm's
int4 layout.

**Experimental (loud warning at load):** `desc_act: true` (act-order) GPTQ and
**AWQ** now decode too, but **to f32** (act-order's `g_idx` groups and AWQ's
nibble order don't fit dlm's flat int4 layout), and their conventions are checked
only by dlm's internal round-trip, *not* against a real export. dlm accepts them
with a warning — verify output with `dlm doctor` and your own prompts. Because
they land as f32, they use ~8× the VRAM of an int4 layer; pair with `--quant int4`
to re-shrink at some accuracy cost.

Still refused:

| refused | why |
| --- | --- |
| `checkpoint_format: gptq_v2` | stores the true zero-point; dlm decodes the classic `zero - 1` convention and has no v2 fixture to check against |
| non-4-bit | dlm decodes 4-bit only |

---

## Distributed & scaling

The Phase 3 serving and scaling components (all CPU-testable, driven by the same
inference engine):

- **OpenAI- and Anthropic-compatible server** ([`src/server`](src/server)) — a
  dependency-free HTTP/1.1 server exposing `/v1/chat/completions`,
  `/v1/completions` and `/v1/models` (OpenAI) plus `/v1/messages` and
  `/v1/messages/count_tokens` (Anthropic), both dialects with `stream: true` SSE.
  Behind it, an `EngineService` runs a background
  batching thread so concurrent requests are **continuously batched** and each can
  be **streamed** to the client token by token. Started by `dlm serve`.
  Per-request **sampling** is honored on both dialects: `temperature`, `top_p`,
  `top_k`, `min_p`, `repetition_penalty` and `seed` select the sampler
  (temperature `0` → deterministic greedy); `stop` (OpenAI) / `stop_sequences`
  (Anthropic) truncate the completion, and `--eos-token` overrides the
  `eos_token_id` auto-detected from `config.json`. **Real tokenizers** load from
  HF `tokenizer.json` (BPE, with special tokens) or `vocab.json` + `merges.txt`,
  and `--chat-template {plain,chatml,llama3}` renders chat messages in the model's
  trained format (control tokens become single ids via the special-token
  vocabulary). Hardening: `--api-key` requires a key on **every** route except
  `/`, `/health` and `/healthz` — including `/metrics`, which leaks request and
  token counts. Send it as either `Authorization: Bearer <key>` (OpenAI) or
  `x-api-key: <key>` (Anthropic), so either SDK authenticates unchanged. The
  request body is size-capped.
  `GET /metrics` exposes Prometheus counters
  (requests, prompt/completion tokens, and — when streaming — layer cache
  hits/misses/evictions/prefetches and live prefetch depth). The engine picks its
  compute kernel from flags:
  - `--stream [--resident-layers N]` — **layer streaming**
    ([`src/forward/streaming.rs`](src/forward/streaming.rs)): only a bounded window
    of layers is held in memory (pinned embedding/LM-head stay resident); the rest
    are materialized on demand from the mmap'd checkpoint through an LRU and evicted
    least-recently-used, so a model can **exceed the resident budget**. The window
    defaults to the VRAM plan's `layers_to_load`. A background worker prefetches the
    next `--prefetch-depth N` layers ahead of compute (`--auto-prefetch` tunes the
    depth from measured load-vs-compute time); `0` disables it. Output is
    bit-for-bit identical to a fully-resident run (tested end-to-end).
    `--ram-cache-gb N` keeps materialized layers in a host-RAM LRU of at most `N`
    GiB, so a layer evicted from the window is not re-read and re-materialized
    from the checkpoint the next time it comes round — roughly **2x** on the
    streamed path. Off by default: the cache duplicates weights in RAM on top of
    the OS page cache, and on a memory-tight box that trade is a loss.
  - `--device gpu` — run the batched engine on the CUDA `GpuKernel`
    (all layers resident in VRAM; requires a `cuda-kernels` build).
  - `--stream --device gpu` — stream a window of layer weights **through VRAM**
    ([`src/forward/streaming_gpu.rs`](src/forward/streaming_gpu.rs)) while KV stays
    resident: run a model larger than VRAM on the GPU. Validated on hardware
    against the CPU oracle ([`tests/gpu_parity.rs`](tests/gpu_parity.rs)); the
    streamed window is bit-comparable to a fully-resident GPU run.
  - `--multi-gpu-ids` — pipeline-parallel across local GPUs (below).

  Two orthogonal memory knobs apply to any kernel:
  - `--quant {int4,int8}` — see [weight precision](#weight-precision---quant) below.
  - `--kv-quant {none,int8,int4}` — quantize the KV cache to int8 (~½ the KV
    memory) or int4 (~¼, more error), trading precision for a longer context in
    the same budget. Defaults to exact `f32`. Independent of `--quant`: one sizes
    the weights, the other the KV history.
  - `--prefix-cache-size N` — cache up to `N` prompt-prefix KV snapshots so
    requests sharing a prefix (e.g. a common system prompt) skip re-prefilling it.
    Each entry holds its prefix's KV in RAM; `0` disables it.

  **Diagnostics.** `dlm doctor` reports the GPU backend and free VRAM, runs a CPU
  inference self-check, and — on a `cuda-kernels` build with a GPU present — runs a
  live CPU-vs-GPU parity probe; pass `--model-path` to check a checkpoint loads and
  tokenizes.

  GPU paths are validated against the CPU oracle in
  [`tests/gpu_parity.rs`](tests/gpu_parity.rs) (resident, streamed, and
  pipeline-parallel kernels, including a realistic `hidden_size` and the Q/K/V-bias
  and RoPE-scaling paths). CI has no GPU, so those tests only run on a machine with
  one — `cargo test --features cuda-kernels` on a CUDA box is the check to run
  before trusting a GPU release; CI type-checks the FFI on every push.
- **Speculative decoding** ([`src/speculative.rs`](src/speculative.rs)) — a cheap
  draft model proposes `gamma` tokens, the target verifies them; accepted tokens
  advance in bulk. With greedy sampling the output is provably **identical** to
  plain target decoding (tested), with acceptance-rate stats. Exposed one-shot
  (`SpeculativeDecoder`) and as a resumable per-round `SpeculativeSession`.
- **Continuous batching** ([`src/batching.rs`](src/batching.rs)) — a scheduler
  keeps up to `max_batch` generations in flight, advancing each one token per
  tick and admitting queued requests as slots free. Each request's output is
  identical to running it alone (tested), independent of interleaving.
  `BatchScheduler::with_speculative` swaps each slot for a `SpeculativeSession`,
  so **the server engine decodes speculatively** when `dlm serve` is given
  `--draft-model-path` — a tick then advances a request by a whole accept/reject
  round, streaming the accepted tokens through the same path. Per-request
  acceptance is surfaced in the OpenAI `usage` response under a `speculative`
  block (`draft_proposed`, `draft_accepted`, `acceptance_rate`); streaming
  responses carry it in a final usage-only chunk.
- **Multi-GPU pipeline parallelism** ([`src/forward/multigpu.rs`](src/forward/multigpu.rs))
  — `dlm serve --multi-gpu-ids 0,1,2` splits the model's layers into contiguous
  per-GPU stages ([`partition_layers`]) and runs each layer on the GPU that owns
  it, calling `cudaSetDevice`/`hipSetDevice` before its block so only the hidden
  residual crosses the inter-GPU boundary (specs §3.3). It's a `ComputeKernel`
  wrapper (`PipelineParallelKernel`), so the batched/speculative server drives a
  multi-GPU model unchanged. Off-GPU `set_device` is a no-op, so a split run is
  **bit-for-bit identical** to a single-device run (tested).
- **Distributed master-worker** ([`src/distributed`](src/distributed)) — layers
  are partitioned into shards across worker nodes; a coordinator streams the
  hidden state through them as **Protobuf** (`prost`) messages, length-prefixed
  over plain TCP. Tensors ride in a packed `repeated float` field (bit-exact
  `f32`, so a distributed forward equals a local one); the framing stays
  synchronous and thread-per-connection rather than the full gRPC/tonic stack.
  **Heartbeats** track liveness and an unreachable worker **falls back to local
  CPU-RAM** execution, so a forward pass still completes. Start a worker with
  `dlm serve --distributed-mode worker`.

[`partition_layers`]: src/distributed/shard.rs

```rust
// Split a model across two worker nodes; the coordinator routes through them.
let shards = dlm::distributed::partition_layers(num_layers, 2);
let mut coord = dlm::distributed::Coordinator::new(cfg, layers, embed, norm, head, vocab, routes)?;
let tokens = coord.generate(&prompt, 32)?;   // == local greedy; survives a dead worker
```

> **Scope note.** These components run and are tested on CPU over localhost. The
> pieces that need real hardware — fused batched/speculative *speedups* (a batch
> kernel), multi-GPU NCCL/RCCL transport, and gRPC/Protobuf instead of the
> hand-rolled TCP protocol — are documented at their call sites; the scheduling,
> protocol, routing, fault-tolerance, and correctness are all implemented and
> verified here.

---

## Contributing & license

- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md) — please report vulnerabilities privately
- [License](LICENSE.md) — Apache-2.0
