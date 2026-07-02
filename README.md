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
- [Building with CUDA](#building-with-cuda)

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
| CUDA runtime FFI (mem-info, host-alloc, streams, async memcpy) | [`src/cuda`](src/cuda) |

The CUDA-specific paths are behind a `cuda` [feature flag](#building-with-cuda);
with it off, the engine uses a page-aligned host fallback with the same layout
contract, so the logic runs on any machine.

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
- **(Optional, for the GPU path)** NVIDIA CUDA Toolkit 12.x with `nvcc` and
  `cudart` on the library path. Not required for building, testing, or the demo.

## Build & run locally

Clone and build (host-only, no GPU needed):

```bash
git clone <your-fork-url> flip
cd flip
cargo build            # debug build
cargo build --release  # optimized build
```

Run the demonstration binary. With no arguments it profiles a representative
Llama-3-70B-class model against a simulated 16 GB card:

```bash
cargo run
```

Example output:

```
flip v0.1.0 — Phase 1 (Local Foundation)
  cuda backend : disabled (host fallback)
  host page    : 4096 bytes

model source : built-in Llama-3-70B-class sample
geometry     : 80 layers, hidden 8192, 64 q-heads / 8 kv-heads, head_dim 128
quantization : Int4 (0.5 bytes/param), ~70.6 B params

free VRAM    : simulated 16 GiB (no CUDA device)

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

swap cycle   : 3 streaming pass(es), window of 29 layer(s)
  ...
pipeline     : 4 steps, 2 overlapped (DMA hidden under compute)
  A:compute —              | B:prefetch p0 → [A]
  A:compute p0 [A]         | B:prefetch p1 → [B]
  ...
```

Point it at a real model directory (containing `config.json` and
`*.safetensors` shards) to map the actual weights and profile from measured
layer sizes:

```bash
cargo run -- /path/to/models/Llama-3-70B-Instruct
```

The storage engine will memory-map the shards, build the layer catalog, and the
profiler will use real per-block byte sizes plus the true Pinned Zone cost.

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

- **`M_free`** — free VRAM from `cudaMemGetInfo()` at runtime (simulated
  off-GPU).
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
├── main.rs           # demonstration binary
├── error.rs          # unified FlipError / Result
├── model/            # config.json parsing, quant schemes, tensor naming
├── storage/          # mmap engine, safetensors parser, layer catalog
├── profiler/         # dynamic VRAM budget math
├── memory/           # page-size discovery + page-locked staging buffers
├── swap/             # linear layer-swap cycle (windows over the model)
├── pipeline/         # double-buffered A/B schedule + host executor
└── cuda/             # feature-gated CUDA runtime FFI + safe wrappers
tests/
└── phase1.rs         # integration tests
build.rs              # links cudart when the `cuda` feature is enabled
```

## Building with CUDA

The GPU path is gated behind the `cuda` Cargo feature. It requires the CUDA
Toolkit with `cudart` reachable by the linker (set `CUDA_PATH` if it lives
outside the default search path). Type-checking works without the toolkit:

```bash
cargo check --features cuda            # validates the FFI, no linking
cargo build --features cuda            # requires cudart on the link path
CUDA_PATH=/usr/local/cuda cargo build --features cuda
```

With the feature on, `PinnedBuffer` allocates genuine page-locked memory via
`cudaHostAlloc`, and `cudaMemGetInfo` reports the live device's free VRAM. With
it off, buffers are page-aligned host allocations (same layout contract,
promotable in place later via `cudaHostRegister`) so nothing about the pipeline
shape changes between builds.

See [`PRD.md`](PRD.md) for product requirements and [`specs.md`](specs.md) for
the full technical specification.
