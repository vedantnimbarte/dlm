# Speed baselines

Measured with `dlm bench` so later changes can be compared against the same
numbers. Each row is the median of 3 runs. Rerun the command shown to reproduce
a row, and put the before/after in the PR that changes it.

To check a change for a speed regression, save a baseline from the old build
with `--json` and compare the new build against it with `--compare`, using the
same flags on the same machine:

```sh
# on the base commit
dlm bench --model-path <model> [flags...] --json base.json
# on the change
dlm bench --model-path <model> [flags...] --json new.json --compare base.json
```

`--compare` prints each metric's change per batch size and exits non-zero if
prefill or decode tok/s fell by more than `--tolerance` percent (default 5).
Latency and memory are shown but never fail the check. The JSON records the
model, quant, flags, GPU, time, and per-batch peak RSS and VRAM; it records the
commit only if the binary was built with `DLM_GIT_COMMIT` set, e.g.
`DLM_GIT_COMMIT=$(git rev-parse --short HEAD) cargo build --release ...`.
Clocks drift between runs, so measure both sides back to back, and rerun before
believing a failure near the tolerance.

**These tables were re-measured on 2026-11-17 on `release/0.5.0` (`b0a612e`),
in one sitting, with the card otherwise idle.** The July 31 numbers (the
development build before 0.4.0) are kept beside them for comparison. Clocks on
this card drift between runs, so only back-to-back comparisons mean much.

## Machine

These numbers come from one machine:

- **GPU:** NVIDIA GeForce GTX 1650, 4 GB (Turing), driver 621.24
- **CPU:** AMD Ryzen 5 5600GT, 6 cores / 12 threads
- **RAM:** 16 GB
- **OS:** Windows 11
- **Build:** `cargo build --release --features cuda-kernels` on `b0a612e`.
  The Gemma 3 rows were re-measured on 2026-11-24 at `c8933e5` (#151), with
  other apps holding 430-470 MiB of the card's VRAM (see the notes under each
  table). The 2026-11-17 Gemma rows used `b0a612e` plus #138, because Gemma
  GGUF files did not load without that PR.

## Resident on GPU

Workload: prompt 512 tokens, 64 generated, `--context-length 1024`, median of
3 runs.

```sh
dlm bench --model-path <model> [--quant ...] --context-length 1024           --prompt-len 512 --gen-len 64 --batch 1,4
```

| Model | Quant | Batch | Prefill tok/s | TTFT ms | Decode tok/s | ms/step | Peak RSS | VRAM in use | July decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | bf16 | 1 | 1,676 | 314 | 124.0 | 8.1 | 2.72 GiB | 1.91 GiB | 38.8 |
| Qwen2.5-0.5B | bf16 | 4 | 1,648 | 319 | 340.6 | 11.7 | 2.72 GiB | 1.91 GiB | 38.7 |
| Qwen2.5-0.5B | int4 | 1 | 1,723 | 304 | 189.5 | 5.3 | 2.58 GiB | 1.31 GiB | 35.6 |
| Qwen2.5-0.5B | int4 | 4 | 1,690 | 311 | 523.6 | 7.6 | 2.58 GiB | 1.31 GiB | 35.1 |
| Gemma 3 4B | GGUF Q4_K_M | 1 | 145 | 3,555 | 30.4 | 32.9 | 9.23 GiB | 3.85 GiB | 6.8 (safetensors int4) |

At batch 4, decode tok/s is the total over all four sequences. VRAM is now
sampled at the end of decode, so it includes the KV caches (July's was sampled
after they were freed). Qwen2.5-1.5B was not re-measured: the checkpoint is no
longer on this machine. Gemma 3 4B now fits without lowering the safety margin.
Its prefill is slow next to Qwen's because the GGUF's Q4_K/Q6_K blocks go
through the GEMV path one token at a time.

The Gemma 3 row was measured on `c8933e5` from the GGUF alone (no `mmproj`
beside it, so no vision tower loads), with other apps holding 430-470 MiB of
VRAM. It nearly matches the 2026-11-17 row, but the two are not a fair
comparison. Before #151 the build filled the card and WDDM paged part of it to
host RAM, so its speed depended on what else was using the GPU:

- **Same sitting:** 0.5.2 (`46b3127`) alternated with `c8933e5` decoded at
  75.9 and 120.0 ms/step against 37.3 and 34.2.
- **Quieter card, earlier on 2026-11-24:** this workload gave 164-170 tok/s
  prefill and 27.7-29.6 ms/step decode with #151's changes.

Batch 4 is left out: its four 576-token sequences need 2,304 KV tokens, and
the pool sized from the VRAM left after the weights held 1,296, so `dlm bench`
stops with `KvCacheExhausted`. With fewer apps holding VRAM, the same build
gave a 2,768-token pool at `--context-length 2048`. Before #151 the pool was
sized from an estimate, so batch 4 ran but spilled to host RAM.

## Streamed through VRAM

Workload: prompt 128 tokens, 16 generated, `--context-length 512`, with
`--breakdown`.

```sh
dlm bench --model-path <model> --stream [--quant ...] --context-length 512           --prompt-len 128 --gen-len 16 --breakdown
```

| Model | Quant | Prefill tok/s | TTFT ms | Decode tok/s | ms/step | Peak RSS | VRAM in use |
|---|---|---|---|---|---|---|---|
| Gemma 3 4B | GGUF Q4_K_M | 166 | 799 | 34.6 | 28.9 | 8.68 GiB | 3.73 GiB |

At Q4_K_M, Gemma 3 4B fits the card whole, so `--stream` keeps all 34 layers
resident and nothing is streamed. In July, at int4 from safetensors, it streamed
at 2.07 tok/s. The first run's prefill (28 tok/s) pays for loading every layer
into the window. Measured on `c8933e5` in the same sitting as the resident row,
with the same 430-470 MiB of VRAM held by other apps. The 2026-11-17 row read
187 / 714 / 34.1 / 29.3 on `b0a612e` plus #138.

### Streamed MoE: TinyMixtral 4x248M

```sh
dlm bench --model-path TinyMixtral-4x248M-MoE --quant int4 --stream --resident-layers 4           --expert-cache-gb <0.05|0.5> --context-length 1024 --prompt-len 256 --gen-len 32           --batch 1,4 --max-batch 4 --runs 3
```

Measured with three builds, each built on `b0a612e`:

- **HEAD:** `b0a612e` itself;
- **no #131:** HEAD with #131 (pinned, staged expert uploads) reverted;
- **no #132:** HEAD with #132 (batched expert rows) reverted.

The builds ran in turn, HEAD first and last. 0.05 GiB of expert cache misses
constantly; 0.5 GiB holds every expert.

| Expert cache | Build | b1 prefill | b1 decode | b4 prefill | b4 decode |
|---|---|---|---|---|---|
| 0.05 GiB | HEAD (first / last) | 400 / 444 | **20.8 / 24.6** | 470 / 464 | **46.4 / 50.4** |
| 0.05 GiB | no #131 | 388 | 11.5 | 394 | 31.7 |
| 0.05 GiB | no #132 | 406 | 10.8 | 376 | 32.6 |
| 0.5 GiB | HEAD (first / last) | 537 / 577 | 29.6 / 32.9 | 592 / 579 | 223 / 224 |
| 0.5 GiB | no #131 | 574 | 29.6 | 599 | 208 |
| 0.5 GiB | no #132 | 550 | 31.9 | 591 | 214 |

With the expert cache missing, each of #131 and #132 roughly doubles batch-1
decode and adds ~50% at batch 4. When every expert stays cached, the three
builds are within noise. Their own PRs measured within noise, because the RAM
tier then still re-read cores from disk every token (fixed in #134), which
hid both.

## CPU

Workload: prompt 128 tokens, 16 generated, `--context-length 512 --device cpu`.

| Model | Quant | Prefill tok/s | TTFT ms | Decode tok/s | ms/step | Peak RSS | July decode tok/s |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | bf16 | 15.4 | 8,372 | 12.21 | 81.9 | 2.86 GiB | 7.38 |

## Batch-1 power (2026-11-17)

Why does single-stream decode draw nearly batch-16 power? `bench/power.py`,
extended to also sample SM and memory clocks and memory-controller
utilization, on Qwen2.5-0.5B: prompt 256 tokens, 256 generated, batch 1.
vLLM 0.33 ran the same workload in WSL, sampled the same way.

| Run | Decode tok/s | Board W | tok/J | GPU util | Memory util | SM / mem clock |
|---|---|---|---|---|---|---|
| dlm bf16 (int8 KV) | 120–124 | 78.6–79.4 | 1.55 | 95% | 80% | 1905 / 6000 MHz |
| dlm bf16, `--kv-quant f16` | 122.5–123.2 | 78.7–78.9 | 1.56 | 95% | 80% | 1906 / 6000 MHz |
| dlm bf16, `--lm-head-int4` | 142.3–142.8 | 78.6–78.7 | 1.81 | 96% | 76% | 1906 / 6000 MHz |
| dlm `--quant int4` | 181–191 | 73.7–76.3 | **2.5** | 85% | 45% | 1919 / 6000 MHz |
| vLLM fp16 | 131.1 | 67.6 | 1.94 | 87% | 35% | 1900 / 5974 MHz |
| idle | | 17.9 | | | | 553 / 584 MHz |

The Windows and WSL `nvidia-smi` give the same readings (checked on one dlm
run sampled from both at once), so the two engines' rows compare directly.

- **What sets batch-1 power is how long the SMs stay busy at full clock, not
  how many bytes move.**
  - The card boosts to 1.9 GHz SM and 6 GHz memory whenever decode runs, for
    every engine and setting.
  - int4 halves the memory controller's busy time against bf16 (45% vs 80%),
    yet still draws 74–76 W, because it keeps the SMs as busy dequantizing.
  - Batch 16 draws 80–81 W: the same clocks, with 97% utilization.
- **vLLM fp16 spends less SM time per token.** Its kernels are busy for 87% of
  a 7.6 ms step; dlm's bf16 kernels are busy for 95% of 8.2 ms. That ~20% more
  busy time per token is most of the 11 W gap. dlm's memory controller is also
  busy over twice as long for the same ~1 GB of weights, which points at the
  bf16 GEMV's access pattern rather than its byte count. Without Nsight
  Compute on this machine, that is as far as counters can go.
- **dlm cannot lower clocks for a memory-bound step.** That would save power
  at little speed cost, but locking clocks needs administrator rights, and
  GeForce cards refuse application clocks. Energy per token falls only as
  kernel time per token falls.
- **For low-traffic serving on this card, use a 4-bit model.** At batch 1,
  `--quant int4` (and GGUF Q4) already gets 2.5 tok/J, 30% better than vLLM
  fp16's 1.94. On a bf16 model, `--lm-head-int4` gives +17% tok/J for the
  +0.025 nats its help text states. The next step for bf16 is profiling its
  GEMV with Nsight Compute.

## What the July numbers said

What the July tables showed, each pointing at a roadmap item. All of them
have since been worked on; `CHANGELOG.md` has the before and after, and
"Against vLLM" below the current comparison.

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

## Against vLLM

Measured on the machine above against vLLM 0.33.0 (torch 2.14+cu132) in WSL2
Ubuntu on the same card. Qwen2.5-0.5B-Instruct, prompt 256 tokens, 128
generated, context 1024, greedy.

```sh
dlm bench --model-path models/Qwen2.5-0.5B-Instruct --context-length 1024 \
          --prompt-len 256 --gen-len 128 --batch 1,4,8,16 --max-batch 16 --runs 3
```

vLLM was driven by `bench/vllm_bench.py` (`python vllm_bench.py float16`, then
`float32`, with the model at `~/models/Qwen2.5-0.5B-Instruct`): `LLM.generate`
on random token ids with `ignore_eos=True` and prefix caching off, median of 3
after a warmup per shape.
Prefill is `batch * 256 / t(1 token)`; decode is `batch * 127 / (t(128) - t(1))`,
so vLLM's prefill includes its per-call Python overhead. Both precisions ran
with `max_num_seqs=16, max_num_batched_tokens=4096, gpu_memory_utilization=0.78`
(fp32 needs them to fit beside the desktop).

### 2026-11-12: cost per token, dlm `a94f12e`

The same workload with the GPU's board power sampled every 100 ms by
`bench/power.py`, which times each batch's work from marker lines in the
benchmark's output so model loading and teardown stay out:

```sh
python bench/power.py "  b" -- dlm bench ... --batch 1,4,8,16 --runs 4   # runs 2-4 of each batch
python bench/power.py BEGIN -- python vllm_bench.py float32 1,4,8,16    # in WSL, beside vLLM
```

Alternating rounds: dlm, vLLM fp32, vLLM fp16 (batch 1), dlm with f16 KV then
int8, vLLM again, dlm. dlm runs its default int8 KV cache. The card's clocks drifted up
through the session, so the same engine decoded up to 40% faster in a later
round; ranges are the spread over rounds, and the two engines' runs next to
each other kept the same order. Energy per token is the mean power while a
batch runs divided by its decode tok/s, the same for both engines.

| Batch | dlm tok/s | dlm W | **dlm tok/J** | vLLM fp32 tok/s | vLLM fp32 W | **vLLM fp32 tok/J** |
|---|---|---|---|---|---|---|
| 1 | 91–129 | 68–83 | 1.33–1.56 | 68–78 | 73–78 | 0.94–1.00 |
| 4 | 267–366 | 71–85 | **3.75–4.31** | 218–256 | 88–89 | 2.48–2.86 |
| 8 | 350–452 | 72–84 | **4.83–5.37** | 297–323 | 89 | 3.34–3.63 |
| 16 | 437–529 | 78–86 | **5.57–6.19** | 322–326 | 89 | 3.64–3.67 |

vLLM fp16 at batch 1: 119–130 tok/s at 70 W, **1.70–1.87 tok/J**.

What a million generated tokens costs at these rates, electricity at an assumed
$0.15/kWh, GPU time to multiply by whatever the card rents for:

| Batch | dlm $/1M | vLLM $/1M | dlm GPU-h/1M | vLLM GPU-h/1M |
|---|---|---|---|---|
| 1 | $0.027–0.031 | $0.022–0.025 (fp16) | 2.15–3.07 | 2.13–2.34 (fp16) |
| 16 | **$0.0067–0.0075** | $0.0113–0.0114 (fp32) | **0.52–0.64** | 0.85–0.86 (fp32) |

- **Batched, dlm costs 25–40% less per token**: at batch 16, 1.5–1.7x the
  tokens per joule and 1.3–1.6x the throughput of vLLM fp32, the best vLLM
  precision for batches on this card. vLLM fp32 runs the
  card at its 90 W limit; dlm sits at 78–86 W.
- **At batch 1, vLLM fp16 is cheaper on energy**: the same speed as dlm's
  later rounds at 13 W less. dlm's single-stream decode draws batch-sized
  power; that is the next thing to look at for low-traffic serving.
- **Electricity is the small part.** At full load the card burns about
  $0.013 of power an hour at $0.15/kWh, so any rental price above that makes
  GPU-hours, that is throughput, the cost that matters.
- dlm's f16 KV cache (`--kv-quant f16`, one round) landed inside the int8
  ranges: 5.85 tok/J at batch 16. The int8 default costs no energy here.

### 2026-11-11: dlm `68bced0` (after the batched-decode, phase 1 and phase 2 work)

Run back to back: vLLM fp16 and fp32, then dlm, then vLLM again (its batch 1
repeated within 0.1 tok/s), with the card idle between workloads. dlm's ranges
are this run and one more clean run earlier the same hour; a third, right
after a build, ran a third slower across the board and is left out.

| Decode tok/s | dlm bf16 | vLLM fp16 | vLLM fp32 |
|---|---|---|---|
| batch 1 | 100–105 | **108** | 58–61 |
| batch 4 | **296–319** | 26 | 189–228 |
| batch 8 | **363–378** | 44–47 | 297–316 |
| batch 16 | **443–445** | 78–88 | 269–318 |
| **Prefill tok/s** | ~1,530–1,615 | ~450–540 | **~1,570–1,765** |

At batch 4-16, decode tok/s is the total over all sequences.

- **Batched decode now leads**: 1.4–1.6x vLLM fp32 at batch 4 and 16, +20-27%
  at batch 8, against the next-best vLLM precision. On 2026-11-10 dlm stopped
  gaining after batch 4 (~175 tok/s) while vLLM fp32 reached ~285.
- **Batch 1 is within 3-8% of vLLM fp16** (100–105 against 108). With n-gram
  speculation (`--speculative auto`, on for the GPU since phase 2), a request
  whose text repeats decodes at up to ~317 tok/s; this table is plain decoding.
- **Prefill trails vLLM fp32 by 5-10%** at 256 tokens. dlm's tiled attention
  gains on long prompts (`CHANGELOG.md`); a 256-token prompt is mostly the
  GEMM.
- **vLLM fp16 is crippled on this card by cuBLAS, not by vLLM.** An fp16
  `F.linear` with 2+ rows runs 10-25x slower than with one row and ~4x slower
  than fp32 (the LM head: 1.9 ms at M=1, 44 ms at M=4). Compare against vLLM
  fp32 for batched decode and prefill, fp16 only for batch 1.
- dlm's int4 and GGUF paths were not compared: vLLM has no Marlin kernels on
  Turing, and its GGUF support is experimental. dlm's quantized formats decode
  faster still (`CHANGELOG.md`: Q4_K_M ~660 tok/s at batch 16).

### 2026-11-10: dlm `6b70197` (before that work)

Run back to back, dlm then vLLM.

| Decode tok/s | dlm bf16 | vLLM fp16 | vLLM fp32 |
|---|---|---|---|
| batch 1 | 84 | **101** | 60 |
| batch 4 | 170 | 22 | **200** |
| batch 8 | 178 | 45 | **281** |
| batch 16 | 173 | 89 | **286** |
| **Prefill tok/s** | ~1,600 | ~450 | ~1,500–1,780 |

dlm ran 84 tok/s at batch 1 here against 108 in `CHANGELOG.md` on another day;
the gap to vLLM fp16 was within clock drift.
