<p align="center">
  <img src="logo.png" alt="flip logo" width="200">
</p>

# flip

**Dynamic layer-streaming inference engine** — run massive LLMs (70B, 405B+) on
consumer GPUs (e.g. 16 GB VRAM) by streaming transformer layers in and out of
VRAM instead of resident-loading the whole model.

Rather than keeping every weight in VRAM, `flip` keeps only a small window of
transformer blocks resident and continuously "flips" the next window in over the
PCIe bus while the GPU computes the current one — trading a bit of speed for the
ability to run models many times larger than the card.

---

## Install

One line — downloads a prebuilt binary for your platform (Linux/macOS, x86-64 or
arm64) and installs it to `~/.local/bin`. No clone, no build, no Rust toolchain:

```sh
curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/Flip/main/install.sh | sh
```

Then:

```sh
flip doctor          # check your machine + run a self-test
flip --help          # subcommands: serve, generate, profile, tokenize, doctor
```

Set `FLIP_INSTALL_DIR` to change the location. Prefer building from source, or
want the GPU build? See [Build & run locally](#build--run-locally) and
[Building for GPU](#building-for-gpu-nvidia--amd) — the prebuilt binary is the CPU
build (runs anywhere); GPU support (`--features cuda-kernels`) is source-only.

Rust users can also `cargo install --git https://github.com/vedantnimbarte/Flip`
(builds from source).

---

## Table of contents

- [Install](#install)
- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Build & run locally](#build--run-locally)
- [Running the tests](#running-the-tests)
- [The VRAM budget math](#the-vram-budget-math)
- [Building for GPU (NVIDIA / AMD)](#building-for-gpu-nvidia--amd)
- [Distributed & scaling](#distributed--scaling)

---

## How it works

`flip` partitions VRAM into three regions:

```
┌────────────────────────────────────────────────────────┐
│                      GPU VRAM                          │
├───────────────────┬───────────────────┬────────────────┤
│   PINNED ZONE     │   STREAMING ZONE  │  CACHE ZONE    │
│  • Embedding Block│  • Double-Buffer A│ • Paged KV     │
│  • LM Head / Norm │  • Double-Buffer B│ • Intermediate │
│  • Draft Model    │   (Asynchronous)  │   Residuals    │
└───────────────────┴───────────────────┴────────────────┘
```

- **Pinned Zone** — embedding, LM head, and norms stay resident permanently
  (moving them each step would thrash the PCIe bus).
- **Streaming Zone** — two buffers (`A`/`B`). While `A` executes on the compute
  stream, `B` DMAs the next window of layers in over PCIe, then they swap.
- **Cache Zone** — PagedAttention KV cache + residual activations.

The data path for a streamed layer:

```
mmap weights ──► CPU-RAM cache ──► pinned staging buffer ──► streaming-zone buffer ──► compute
   (NVMe)         (hot layers)       (page-locked host)          (VRAM)
```

Memory-mapping skips the OS read-buffer copy; the tiered CPU-RAM cache keeps hot
layers resident across token steps so they skip the disk read; and the
page-locked (pinned) host buffer lets the PCIe controller DMA straight to VRAM
asynchronously, so disk I/O and copies hide under GPU compute.

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
- **(Optional, for the GPU path)** one of:
  - NVIDIA CUDA Toolkit 12.x with `cudart` on the library path, or
  - AMD ROCm 6.x with `amdhip64` on the library path.

  Not required for building, testing, or the demo — the host fallback needs no GPU.

## Build & run locally

Clone and build (host-only, no GPU needed):

```bash
git clone <your-fork-url> flip
cd flip
cargo build            # debug build
cargo build --release  # optimized build
```

The binary exposes these subcommands:

```bash
cargo run -- --help          # top-level help
cargo run -- profile         # profile a sample 70B-class model (no GPU needed)
cargo run -- serve --help    # full serve flag list (specs §4)
cargo run -- generate --help # end-to-end CPU generation on a synthetic model
cargo run -- tokenize --help # byte-level BPE encode/decode round-trip
cargo run -- doctor          # check machine (GPU/VRAM) + run an inference self-test
```

**`search` / `pull`** — find and download models straight from the
[Hugging Face hub](https://huggingface.co), no `hf` CLI or manual file-grabbing
needed. `pull` shells out to `curl` (built into Linux, macOS, and Windows 10/11)
to fetch only the files flip loads (`config.json`, `*.safetensors`, tokenizer):

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
flip v0.1.0
  gpu backend  : none (host fallback)
  host page    : 4096 bytes

model source : built-in Llama-3-70B-class sample
geometry     : 80 layers, hidden 8192, 64 q-heads / 8 kv-heads, head_dim 128
quantization : Int4 (0.5 bytes/param), ~70.6 B params

free VRAM    : simulated 16 GiB (no GPU device)

── VRAM PLAN ─────────────────────────────────
  M_free           :    16384.0 MiB
  M_safety         :     1536.0 MiB
  M_kv_total       :     2560.0 MiB
  pinned_zone      :        0.0 MiB
  M_layer_weight   :      420.5 MiB
  usable           :    12288.0 MiB
  ▶ layers_to_load :         29 / 80
  ▶ resident       :      36.2%
──────────────────────────────────────────────

kv cache     : 512 paged blocks × 16 tok, 5.00 MiB/block → 8192 token capacity
swap cycle   : 3 streaming pass(es), window of 29 layer(s)
pipeline     : 4 steps, 2 overlapped (DMA hidden under compute)
```

Point it at a real model directory (containing `config.json` and
`*.safetensors` shards) to map the actual weights and profile from measured
layer sizes:

```bash
cargo run -- profile --model-path /path/to/models/Llama-3-70B-Instruct
```

**`serve`** — starts the **OpenAI-compatible HTTP API server** for a model. It
exposes `POST /v1/chat/completions`, `POST /v1/completions`, and `GET /v1/models`
so clients like Open WebUI can talk to `flip` unchanged:

```bash
cargo run -- serve \
    --model-path /path/to/small-model \
    --context-length 8192 \
    --port 8000 \
    --host 127.0.0.1

# then, from another shell (add "stream": true for token-by-token SSE):
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"flip","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

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

By default generation runs on the CPU. On a `cuda-kernels` build, `--device gpu`
runs the same model through `GpuKernel` instead (the CLI errors with a clear
message if the binary wasn't built with the feature):

```bash
cargo run --features cuda-kernels -- generate --model-path /path/to/small-model --device gpu
```

The standalone `tokenize` subcommand shows the encoder round-trip:

```bash
cargo run -- tokenize --text "Hello, world!"
# ids        : [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
# round-trip : "Hello, world!" (ok)
```

The loader handles both float (`.weight` in F32/F16/BF16) and **GPTQ-style 4-bit
quantized** (`.qweight`/`.qzeros`/`.scales`) projections, dequantizing the latter
into dense weights on load. (The int32 packing, grouping, and transpose are
round-trip-tested; matching a specific exporter byte-for-byte — AWQ's nibble
permutation, GPTQ's zero-point bias — would need real fixtures, noted in
[`src/quant/packed.rs`](src/quant/packed.rs).)

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

| Feature | Vendor | Runtime | Env var |
|---|---|---|---|
| `cuda` | NVIDIA | `cudart` | `CUDA_PATH` |
| `cuda-kernels` | NVIDIA | `cudart` + compiled `kernels.cu` (nvcc) | `CUDA_PATH` |
| `rocm` | AMD | `amdhip64` | `ROCM_PATH` |
| _(none)_ | — | host fallback | — |

`cuda-kernels` additionally compiles [`src/gpu/kernels.cu`](src/gpu/kernels.cu)
with nvcc and enables `GpuKernel` (the device `run_block`). `cargo check
--features cuda-kernels` type-checks the Rust FFI without the toolkit; building a
binary that runs the kernels needs nvcc + a GPU.

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

## Distributed & scaling

The Phase 3 serving and scaling components (all CPU-testable, driven by the same
inference engine):

- **OpenAI-compatible server** ([`src/server`](src/server)) — a dependency-free
  HTTP/1.1 server exposing `/v1/chat/completions` (with `stream: true` SSE),
  `/v1/completions`, `/v1/models`. Behind it, an `EngineService` runs a background
  batching thread so concurrent requests are **continuously batched** and each can
  be **streamed** to the client token by token. Started by `flip serve`.
  Per-request **sampling** is honored: `temperature`, `top_p`, `top_k`, and `seed`
  select temperature/nucleus/top-k sampling (temperature `0` → deterministic
  greedy); `stop` sequences truncate the completion. **Real tokenizers** load from
  HF `tokenizer.json` (BPE, with special tokens) or `vocab.json` + `merges.txt`,
  and `--chat-template {plain,chatml,llama3}` renders chat messages in the model's
  trained format (control tokens become single ids via the special-token
  vocabulary). Hardening: `--api-key` requires a bearer token on `/v1/*`, and the
  request body is size-capped. The engine picks its compute kernel from flags:
  - `--stream [--resident-layers N]` — **layer streaming**
    ([`src/forward/streaming.rs`](src/forward/streaming.rs)): only a bounded window
    of layers is held in memory (pinned embedding/LM-head stay resident); the rest
    are materialized on demand from the mmap'd checkpoint through an LRU and evicted
    least-recently-used, so a model can **exceed the resident budget**. The window
    defaults to the VRAM plan's `layers_to_load`. Output is bit-for-bit identical to
    a fully-resident run (tested end-to-end).
  - `--device gpu` — run the batched engine on the CUDA `GpuKernel`
    (all layers resident in VRAM; requires a `cuda-kernels` build).
  - `--stream --device gpu` — stream a window of layer weights **through VRAM**
    ([`src/forward/streaming_gpu.rs`](src/forward/streaming_gpu.rs)) while KV stays
    resident: run a model larger than VRAM on the GPU. **Experimental —
    compile-checked but not yet validated on real GPU hardware.**
  - `--multi-gpu-ids` — pipeline-parallel across local GPUs (below).

  **Diagnostics.** `flip doctor` reports the GPU backend and free VRAM, runs a CPU
  inference self-check, and — on a `cuda-kernels` build with a GPU present — runs a
  live CPU-vs-GPU parity probe; pass `--model-path` to check a checkpoint loads and
  tokenizes. This is how you validate the GPU paths on real hardware (there is no
  GPU in CI, so `--device gpu`, `--stream --device gpu`, and `--multi-gpu-ids` are
  compile-checked only until run through `flip doctor` / a real serve on a GPU box).
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
  so **the server engine decodes speculatively** when `flip serve` is given
  `--draft-model-path` — a tick then advances a request by a whole accept/reject
  round, streaming the accepted tokens through the same path. Per-request
  acceptance is surfaced in the OpenAI `usage` response under a `speculative`
  block (`draft_proposed`, `draft_accepted`, `acceptance_rate`); streaming
  responses carry it in a final usage-only chunk.
- **Multi-GPU pipeline parallelism** ([`src/forward/multigpu.rs`](src/forward/multigpu.rs))
  — `flip serve --multi-gpu-ids 0,1,2` splits the model's layers into contiguous
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
  `flip serve --distributed-mode worker`.

[`partition_layers`]: src/distributed/shard.rs

```rust
// Split a model across two worker nodes; the coordinator routes through them.
let shards = flip::distributed::partition_layers(num_layers, 2);
let mut coord = flip::distributed::Coordinator::new(cfg, layers, embed, norm, head, vocab, routes)?;
let tokens = coord.generate(&prompt, 32)?;   // == local greedy; survives a dead worker
```

> **Scope note.** These components run and are tested on CPU over localhost. The
> pieces that need real hardware — fused batched/speculative *speedups* (a batch
> kernel), multi-GPU NCCL/RCCL transport, and gRPC/Protobuf instead of the
> hand-rolled TCP protocol — are documented at their call sites; the scheduling,
> protocol, routing, fault-tolerance, and correctness are all implemented and
> verified here.
