# flip

**Dynamic layer-streaming inference engine** — run massive LLMs (70B, 405B+) on
consumer GPUs (e.g. 16 GB VRAM) by streaming transformer layers in and out of
VRAM instead of resident-loading the whole model.

Rather than keeping every weight in VRAM, `flip` keeps only a small window of
transformer blocks resident and continuously "flips" the next window in over the
PCIe bus while the GPU computes the current one — trading a bit of speed for the
ability to run models many times larger than the card.

---

## Table of contents

- [How it works](#how-it-works)
- [Components](#components)
- [Prerequisites](#prerequisites)
- [Build & run locally](#build--run-locally)
- [Running the tests](#running-the-tests)
- [The VRAM budget math](#the-vram-budget-math)
- [Project layout](#project-layout)
- [Building for GPU (NVIDIA / AMD)](#building-for-gpu-nvidia--amd)

---

## How it works

`flip` partitions VRAM into three regions (see `specs.md` §2):

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
mmap weights ──► pinned staging buffer ──► streaming-zone buffer ──► compute
   (NVMe)         (page-locked host)          (VRAM)
```

Memory-mapping skips the OS read-buffer copy; the page-locked (pinned) host
buffer lets the PCIe controller DMA straight to VRAM asynchronously, so disk I/O
and copies hide under GPU compute.

## Components

| Component | Module |
|---|---|
| Memory-mapped, zero-copy safetensors reader | [`src/storage`](src/storage) |
| Sharded checkpoint support + tensor index | [`src/storage/mmap_store.rs`](src/storage/mmap_store.rs) |
| Layer catalog (real per-layer + pinned byte sizes) | [`src/storage/catalog.rs`](src/storage/catalog.rs) |
| `config.json` geometry + quantization parsing | [`src/model`](src/model) |
| Tensor role classification (pinned vs. streamed) | [`src/model/naming.rs`](src/model/naming.rs) |
| Dynamic VRAM profiling math | [`src/profiler`](src/profiler) |
| Page-locked host staging buffers | [`src/memory`](src/memory) |
| Linear layer-swap cycle | [`src/swap`](src/swap) |
| Double-buffered A/B streaming schedule + host executor | [`src/pipeline`](src/pipeline) |
| PagedAttention block-paged KV cache | [`src/cache`](src/cache) |
| `clap` CLI — `serve` / `profile` subcommands | [`src/cli.rs`](src/cli.rs) |
| GPU runtime FFI — CUDA + ROCm/HIP (mem-info, host-alloc, streams, async memcpy) | [`src/gpu`](src/gpu) |

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

The binary exposes two subcommands:

```bash
cargo run -- --help          # top-level help
cargo run -- profile         # profile a sample 70B-class model (no GPU needed)
cargo run -- serve --help    # full serve flag list (specs §4)
```

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

**`serve`** — resolves the full serving configuration (specs §4) and runs the
planning pipeline. The OpenAI-compatible server loop itself is a later milestone;
today this validates the config and prepares the engine end-to-end:

```bash
cargo run -- serve \
    --model-path /path/to/models/Llama-3-70B-Instruct \
    --vram-budget-gb 13.5 \
    --context-length 8192 \
    --port 8000 \
    --host 127.0.0.1
```

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

The profiler decides how many transformer blocks fit resident at once
(`specs.md` §3.1):

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

## Project layout

```
src/
├── lib.rs            # crate root & public API re-exports
├── main.rs           # CLI entry point (serve / profile dispatch)
├── cli.rs            # clap argument definitions
├── error.rs          # unified FlipError / Result
├── model/            # config.json parsing, quant schemes, tensor naming
├── storage/          # mmap engine, safetensors parser, layer catalog
├── profiler/         # dynamic VRAM budget math
├── memory/           # page-size discovery + page-locked staging buffers
├── swap/             # linear layer-swap cycle (windows over the model)
├── pipeline/         # double-buffered A/B schedule + host executor
├── cache/            # PagedAttention block-paged KV cache
└── gpu/              # vendor-neutral backend: CUDA + ROCm/HIP FFI + wrappers
tests/
└── phase1.rs         # integration tests
build.rs              # links cudart / amdhip64 for the selected GPU feature
```

## Building for GPU (NVIDIA / AMD)

The GPU path is vendor-neutral behind [`src/gpu`](src/gpu), selected by a Cargo
feature. Everything above the backend (storage, profiler, pipeline) is identical
across vendors.

| Feature | Vendor | Runtime | Env var |
|---|---|---|---|
| `cuda` | NVIDIA | `cudart` | `CUDA_PATH` |
| `rocm` | AMD | `amdhip64` | `ROCM_PATH` |
| _(none)_ | — | host fallback | — |

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

See [`PRD.md`](PRD.md) for product requirements and [`specs.md`](specs.md) for
the full technical specification.
