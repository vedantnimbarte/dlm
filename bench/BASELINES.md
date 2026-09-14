# Speed baselines

Measured with `dlm bench` so later changes can be compared against the same
numbers. Each row is the median of 3 runs. Rerun the command shown to reproduce
a row, and put the before/after in the PR that changes it.

## Machine

These numbers come from one machine:

- **GPU:** NVIDIA GeForce GTX 1650, 4 GB (Turing), driver 591.86
- **CPU:** AMD Ryzen 5 5600GT, 6 cores / 12 threads
- **RAM:** 16 GB
- **OS:** Windows 11
- **Build:** `cargo build --release --features cuda-kernels`, on commit `6259ffd`
  plus the `dlm bench` change

## Resident on GPU

Workload: prompt 512 tokens, 64 generated, `--context-length 1024`.

```sh
dlm bench --model-path <model> [--quant ...] --context-length 1024 \
          --prompt-len 512 --gen-len 64 --batch 1,4
```

| Model | Quant | Batch | Prefill tok/s | TTFT ms | Decode tok/s | ms/step | Peak RSS | VRAM in use |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | fp16 | 1 | 77.7 | 6,615 | 38.80 | 25.8 | 2.71 GiB | 1.84 GiB |
| Qwen2.5-0.5B | fp16 | 4 | 78.2 | 6,545 | 38.67 | 103.4 | 2.71 GiB | 1.84 GiB |
| Qwen2.5-0.5B | int4 | 1 | 70.1 | 7,326 | 35.63 | 28.1 | 2.31 GiB | 1.23 GiB |
| Qwen2.5-0.5B | int4 | 4 | 70.2 | 7,250 | 35.10 | 113.9 | 2.31 GiB | 1.23 GiB |
| Qwen2.5-1.5B | int8 | 1 | 22.6 | 22,702 | 14.60 | 68.5 | 5.31 GiB | 2.41 GiB |
| Qwen2.5-1.5B | int4 | 1 | 22.4 | 22,917 | 14.41 | 69.4 | 5.41 GiB | 1.78 GiB |
| Gemma 3 4B | int4 | 1 | 10.2 | 50,519 | 6.80 | 147.0 | 7.40 GiB | 3.51 GiB |

At batch 4, decode tok/s is the total over all four sequences.

The VRAM figures here were sampled after the run, when the KV caches had
already been freed, so they show the weights and scratch buffers without KV.
`dlm bench` now samples at the end of decode.

Gemma 3 4B needs `--safety-margin-gb 0.5` to fit. Under the default 1.5 GiB
margin, the fit check refuses it: it needs 3.8 GiB and only 3.2 GiB is free.

## Streamed through VRAM

Workload: prompt 128 tokens, 16 generated, `--context-length 512`, with
`--breakdown`.

```sh
dlm bench --model-path <model> --stream [--quant ...] --context-length 512 \
          --prompt-len 128 --gen-len 16 --breakdown
```

| Model | Quant | Prefill tok/s | TTFT ms | Decode tok/s | ms/step | PinStage ms/tok | H2D ms/tok | Peak RSS | VRAM in use |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | bf16 | 25.4 | 5,399 | 2.88 | 346.7 | 174.9 | 122.1 | 6.49 GiB | 2.58 GiB |
| Gemma 3 4B | int4 | 10.9 | 12,293 | 2.07 | 482.3 | 241.0 | 153.2 | 9.06 GiB | 1.98 GiB |

The GPU streaming path emits no `Compute` event, so the breakdown has no compute
column. Compute is roughly ms/step minus the two transfer columns. Every layer
was served from the host RAM cache (`RamHit`), so no time went to disk reads.

## CPU

Workload: prompt 128 tokens, 16 generated, `--context-length 512 --device cpu`.

| Model | Quant | Prefill tok/s | TTFT ms | Decode tok/s | ms/step | Peak RSS |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | bf16 | 9.1 | 14,233 | 7.38 | 135.5 | 2.67 GiB |

A first CPU run recorded 1.3 prefill / 0.75 decode tok/s. It did not reproduce:
the same binary, rerun on an idle machine, gave the row above. Something else
was using the CPU during that run. Rerun any number that looks wrong before
trusting it.

## What the numbers say

Each observation points at an item in the roadmap:

- **Batching adds no throughput.** Batch 4 decodes at the same total tok/s as
  batch 1, because the scheduler steps sessions one after another. Fusing the
  batch into one step is the fix.
- **Prefill is barely faster than decode.** A 512-token prompt waits 6.6 s on a
  0.5B model, because GPU prefill runs at most 16 tokens per kernel call.
  Matrix-matrix kernels are the fix.
- **int4 is not faster than fp16 at 0.5B.** The kernel dequantizes per element.
  Group-dequant kernels are the fix.
- **Peak host RAM is 2–7 GiB for models resident on the GPU.** It is peak
  memory, and it builds up while loading, not while decoding. The mmap'd
  checkpoint pages, the per-layer host copies made before upload, and the f32
  embedding table are all resident at once. The host copy of the KV cache is
  not the cause: it is a few tens of MiB at this context, and removing it left
  peak RSS unchanged.
- **Streaming spends most of each step moving bytes.** PinStage + H2D account for
  about 85% of a streamed Qwen 1.5B step, and PinStage alone costs more than the
  PCIe copy. Copying straight from the RAM cache into pinned memory is the fix.
- **The CPU path starts new OS threads for every large matmul** (the MLP
  projections and the LM head), and has no SIMD.
