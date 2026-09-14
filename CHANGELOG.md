# Changelog

All notable changes to dlm are recorded here.

This project's value proposition is being precise about what works, so each
release carries a **Verified** and a **Known gaps** section alongside the usual
changes. `RELEASING.md` holds the full ledger; this file records the delta
between releases, which is the part a user upgrading actually needs.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The VRAM planner sized the KV cache at half its real size.**
  - **Cause:** it assumed 2 bytes per element, as for an fp16 cache, but the
    device KV cache is f32. `--kv-quant` changes only the host-side stores.
  - **Effect:** a plan that claimed to fit could run out of VRAM once the cache
    filled. The fit checks for `serve` and `bench` now reserve the real amount.
  - **Unchanged:** the paged pool's token capacity. Its budget comes from the
    same plan, so the pool gets the same number of tokens.
- **Multi-GPU stages shared one device's scratch memory.** The CUDA kernels'
  scratch buffers were kept per thread, on the assumption that a multi-GPU
  pipeline runs one thread per device. It does not: every stage runs on the
  inference thread, switching devices per layer, so later stages were handed
  buffers allocated on the first stage's GPU. Scratch is now kept per thread and
  per device. Untested across two real GPUs.
- **Host `--stream` ignored per-layer attention windows.** The streamed CPU
  kernel handed every layer the model-wide config, so Gemma 2's global layers
  were windowed too — invisible until a context passed 4,096 tokens.
- **GPT-2 with `--stream` produced garbage** (`" labor labor labor…"`), and
  Falcon's streamed and GPU generators had the wrong final norm. Only the CPU
  resident builder attached GPT-2's learned position embeddings and the
  LayerNorm head (with its bias); every other generator ran without position
  information and with an RMSNorm head. All builders now load and attach them.
- **`dlm pull` failed on every repo.** Since 0.4.0 moved the token out of
  `argv`, curl was spawned with its output inherited rather than captured, so
  the model-info request returned an empty body and the pull stopped with
  "could not read model info … private/gated?".

### Changed

- **CPU matmuls reuse a persistent thread pool.** Large GEMVs (MLP projections,
  the LM head) used to start a fresh OS thread per chunk on every call. On
  Qwen2.5-0.5B on CPU, measured against the previous binary back to back,
  decode went from 7.38 to 8.28 tok/s and prefill from 9.1 to 10.8 tok/s.
- **Sampling no longer sorts the whole vocabulary.**
  - **`top_k`:** only the best k tokens are put in order.
  - **Default `top_k = 0` with `top_p < 1`:** the nucleus is selected before it
    is sorted.
  - **Speed:** one sampled token from Qwen's 152k vocabulary took about 7 ms and
    now takes 1.7 ms on peaked logits (0.7 ms with top-k 40).
  - **Same output:** the distribution is identical to a full sort, and a test
    pins it. Tied logits now order by token id, so a seed draws the same token
    every run.
- **Streamed layers are uploaded into the evicted layer's VRAM buffers.** The
  eviction happens before the upload, not after, and a steady-state miss
  allocates and frees nothing.
  - **Allocations:** over a streamed Qwen2.5-1.5B run, `cudaMalloc` calls
    dropped from 2,296 to 181 and time spent in `cudaMalloc` from 419 to 53 ms.
  - **Peak VRAM:** the window no longer holds an extra layer while one loads.
- **GPU paths no longer mirror the KV cache on the host.** Each layer and token
  appended a zero-filled host row that nothing read. Now only a count is kept.
  At the contexts measured this was tens of MiB, so peak RSS did not move.
  Peak RSS on GPU runs comes from loading, not decode.
- **`--stream` caches layers in host RAM by default, quantized or not**, when the
  whole layer set (plus 25%) fits under a quarter of physical RAM. It was on
  only with `--quant`; unquantized streaming re-read every layer from the mmap on
  every window miss, which took streamed Gemma 3 1B from 46 s to 268 s on a
  1,500-token prompt. When the set does not fit, the cache is now off rather
  than partial (a smaller LRU never hits on a cyclic scan), and serve says so.
  `--ram-cache-gb` still overrides either way.

### Added

- **`--kv-quant f16`: the GPU keeps the KV cache in fp16.**
  - **Before:** device KV was always f32, and `--kv-quant` had no effect on it.
  - **Now:** `f16`, `int8` and `int4` all store device KV as fp16, halving KV
    VRAM. The fit checks reserve accordingly.
  - **Measured:** Qwen2.5-0.5B, 4 sequences at an 8k context: 2.35 GiB -> 1.98
    GiB VRAM in use, with decode speed unchanged.
  - **Output:** greedy output matched f32 KV for 96 of 96 tokens on
    Qwen2.5-0.5B and on Gemma 3 1B.
  - **Unchanged:** the default is still exact f32, MLA stays f32, and the CPU
    kernels keep their f32/int8/int4 stores.
- **`dlm bench`**, a speed harness.
  - **Loading:** it builds the model through `serve`'s own loading path, so
    every `serve` flag that shapes the model applies.
  - **Workload:** a fixed greedy prompt/decode run at one or more batch sizes.
  - **Output:** prefill tok/s, time to first token, decode tok/s and ms per
    step, as the median over `--runs`. It also reports peak RSS, VRAM in use,
    and optional JSON.
  - **`--breakdown`:** splits streamed decode time by pipeline stage.
  - **Reason:** until now no speed claim could be reproduced with one command.
  - **Baselines:** the GTX 1650 numbers are in `bench/BASELINES.md`.
- **`generate` prints prefill and decode speed**, and samples with
  `--temperature`/`--top-p`/`--top-k`/`--seed` instead of decoding greedily only.
- **Gemma 3** (the text-only 270M and 1B). Per-head `(1 + w)` Q/K norms, the
  5:1 local/global layer pattern — read from `sliding_window_pattern`,
  `_sliding_window_pattern`, or a `layer_types` list (an irregular list is
  refused) — and a separate RoPE base for the windowed layers, with
  `rope_scaling` applied to the global layers only. On `gemma-3-1b-it` it answers
  correctly on CPU and GPU, recalls a code from 1,500 tokens back, and scores a
  1,126-token passage at 1.80 nats/token (0.33 on the half only recall can
  predict); the local layers on the global RoPE base score 3.18, and Gemma 2's
  layer rule 3.35.
- **Gemma 3 4B, 12B and 27B, as text models.** These are multimodal checkpoints:
  dlm reads the language model out of `text_config` — filling in
  `Gemma3TextConfig`'s defaults for what Google's sparse exports omit — and out
  of `language_model.*` / `model.language_model.*` tensors, and drops the vision
  tower, whose encoder blocks would otherwise be counted as transformer layers.
  Images are not supported. On `gemma-3-4b-it` (8 GB bf16, streamed through a
  4 GB card) it answers correctly, recalls a code from 1,500 tokens back at
  `--quant int4`, and scores the 1,126-token passage at 1.53 nats/token in bf16.
- `Generator::score`: the per-token log-probability of a sequence under teacher
  forcing, for checking a forward pass against a text's expected cross-entropy.
- **Prompt prefill runs layer by layer on the streamed host path.** A prompt
  used to go through the model one token at a time, so with `--stream` every
  prompt token re-streamed the whole window; now each layer loads once per
  prompt (in chunks of 512 tokens). A 204-token prompt on Qwen2.5-0.5B with
  `--stream --resident-layers 4` prefills in 26 s instead of 56 s — the same as
  the fully-resident run. Output is bit-identical. Kernels gain a stack-level
  `ComputeKernel::prefill`, whose default keeps the old order.
- **Batched prefill on the GPU.** Dense layers now prefill up to 16 prompt
  tokens per device call on both GPU kernels, reusing the batched decode block
  with every slot pointed at the one sequence's KV — which is exact causal
  attention, since each slot writes its row before attending over the rows
  below it. The streaming GPU kernel also goes layer by layer (MLA and MoE
  layers one token at a time), so a layer streams into VRAM once per prompt
  chunk instead of once per token. On a GTX 1650 with Qwen2.5-0.5B, a 204-token
  prompt plus 8 generated tokens with `--stream --resident-layers 4` fell from
  140 s to 10.5 s; the fully-resident prefill went from 8.4 s to 6.5 s. Output is
  unchanged, and six new parity tests pin dense, Gemma2, MoE and MLA+MoE prefill
  against the CPU oracle.
- **GPT-2 and Falcon run on the GPU**, resident and streamed. The device block
  gains LayerNorm (with bias), the ungated MLP, output-projection and MLP
  biases, a no-RoPE mode, and Falcon's parallel residual, selected per layer
  from the config. Previously `--device gpu` ran these models through the
  Llama-shaped kernel and produced garbage without an error. GPT-2 answers
  identically on CPU and GPU (0.36 s vs 2.5 s for 12 tokens on a GTX 1650);
  Falcon is checked against the CPU oracle on synthetic weights, since no
  runnable Falcon checkpoint fits the card.
- **The LM head runs on the GPU.** Every GPU decode step used to finish with
  the vocabulary-wide GEMV on the host — 233M multiply-adds per token for
  Qwen2.5-1.5B — which cost more than the model's entire layer stack on the
  device. The head now uploads with the layers: bf16 when its values are exactly
  bf16 (lossless), f32 otherwise, int8 when the layers were quantized with
  `--quant`. If it does not fit, dlm warns and keeps the host head. On a GTX
  1650, Qwen2.5-1.5B `--quant int8` generates 128 tokens in 10.2 s instead of
  19.7 s; streamed with 8 of 28 layers resident, 50.5 s instead of 56.6 s.
  Multi-GPU pipelines place it on the last stage's GPU, which it selects itself
  before each launch.
- **Speculative decoding samples, and keeps its KV cache.** With
  `--draft-model-path`, each verification used to rebuild the target's KV
  cache and re-run the whole sequence for every token it checked, and request
  sampling parameters were silently ignored (always greedy). The target now
  scores the draft's proposals in one prefill, both models roll their caches back
  to the accepted tokens, and acceptance uses the standard rejection rule, so
  `temperature`, `top_p`, `top_k`, `min_p` and `repetition_penalty` are honored
  and the output follows the target's distribution (greedy stays identical to
  plain decoding). Qwen2.5-1.5B with a 0.5B draft reproduces plain greedy output
  token for token at 80% acceptance, in 13.7 s for 128 tokens where the old
  implementation took 691 s. On the GTX 1650 that is still slower than plain
  decoding (10.2 s): per-layer launch overhead, not weight size, dominates there,
  so a 24-layer draft costs nearly as much per token as a 28-layer target.
- **GPU attention no longer slows down linearly with context.** The attention
  kernel ran one thread per head, each walking the whole history in a scalar
  loop; it is now three launches parallel over (head, position) and (head,
  dim). On a GTX 1650 with Qwen2.5-0.5B at ~600 tokens of context, decode went
  from 254 to 97 ms/token and a 604-token prefill from 40 s to 11 s.
- **MLA attention (DeepSeek) factors through latent space on the GPU.** It ran
  one thread per head and rebuilt every cached position's K and V from the
  latent through `kv_b` for every token. Since K and V are linear in the latent,
  scores now use the query projected into latent space once, and the context
  applies `kv_b`'s value rows once to the weighted latent mix, each parallel
  over (head, position) or (head, latent). At DeepSeek-V2-Lite's dimensions one
  MLA layer decodes in 1.3–2.1 ms/token over 1–1,024 positions on a GTX 1650,
  against 913 ms/token (1–256) and 2,812 ms/token (257–512) before.
- **Chat template auto-detection.** `--chat-template` now defaults to `auto`,
  which fingerprints the checkpoint's Jinja template and picks the matching
  built-in format. Five formats are new — `llama2`, `mistral`, `gemma`,
  `phi3`, `deepseek` join `plain`, `chatml`, `llama3` — and the format's
  end-of-turn token (`<end_of_turn>`, `<|end|>`, …) is added to the stop set.
  Previously every model defaulted to `plain`, a format no instruct model is
  trained on, unless the user knew to pass the flag. Checked against the real
  `tokenizer_config.json` of all thirteen fixture families.

- **Per-layer flow telemetry.** `dlm serve --telemetry` exposes `GET
  /v1/telemetry`, a Server-Sent Events stream of timestamped per-layer events
  (mmap read, RAM-cache hit/miss, staging, H2D copy, compute, eviction,
  prefetch), so a streamed run can answer "where did the time go?" rather than
  only "is the window working?". H2D copies are timed by device events
  (`cudaEventElapsedTime` / `hipEventElapsedTime`), not wall-clock around an
  enqueue; an event without a device timing is flagged estimated. Collection
  runs only while a subscriber is attached, events carry timings, byte counts
  and layer indices only — never prompt or completion text — and the route sits
  behind `--api-key` like `/metrics`. The ring is bounded and drop-oldest, with a
  cumulative loss count.
- `hub::pull_with_progress`, reporting download and SHA-256 verification
  progress to a callback.

## [0.4.0] - 2026-07-31

### Verified

Three model families landed and they are **not** equally proven; `RELEASING.md`
carries the full ledger:

- **Phi-3** — weight mapping proven by a same-weights-twice equivalence test,
  confirmed by mutation.
- **GPT-2** — answers correctly on the real `openai-community/gpt2`.
- **Falcon** — **config-verified only.** No runnable checkpoint exists: the
  smallest `alibi: false` variant is 14 GB, and the 1B `falcon-rw` uses ALiBi,
  which dlm refuses. Not yet output-verified.

GPU paths are unchanged and remain verified only on Turing (GTX 1650); CI has no
GPU and never executes a device kernel.


### Fixed

- **A panicking request handler leaked a connection slot permanently.** The live
  connection counter was decremented on the last line of the connection thread,
  which a panic unwinds past. After 256 cumulative panics the server shed every
  connection forever, with no log line and a process that still looked healthy.
- **`/healthz` was exempt from authentication but routed nowhere**, so a
  Kubernetes liveness probe pointed at it got a 404 and restarted a healthy
  process. Fixed on both the batched and distributed routers.
- **Query strings broke routing and the auth exemption.** `/health?probe=1` was
  not recognised as the health route and returned `401` under `--api-key`;
  `/v1/models?limit=1` returned `404`.
- **The Hugging Face token was passed to `curl` in `argv`**, where
  `/proc/<pid>/cmdline` exposed it to any local user for the duration of a pull.
  It now goes through a curl config file on stdin.
- **`dlm pull` did not validate the repo id**, only counted its segments, so
  `a/..` resolved the download directory to the *parent* of the intended one.
- Connections past the concurrency cap are now refused with `503` and a
  `Retry-After` instead of being dropped silently, which was indistinguishable
  from a crashed server.

### Added

- **HTTP keep-alive.** Responses with a `Content-Length` now leave the connection
  open, so the pooling OpenAI and Anthropic SDKs stop paying a TCP handshake per
  request. `Connection: close` is honoured, HTTP/1.0 defaults to closing,
  streaming (SSE) responses always close, and a connection is recycled after 100
  requests so one client cannot hold a thread indefinitely.
- **`dlm pull` verifies downloaded weights against the hub's published SHA-256.**
  Previously only the byte count was checked, which catches truncation and
  nothing else. A file that fails is deleted, so the next `pull` refetches it
  rather than treating it as already complete.
- **Falcon support** (config-verified). Parallel attention/FFN, multi-query,
  LayerNorm, ungated MLP, fused `query_key_value`. ALiBi variants and
  Falcon-40B's interleaved layout are refused rather than mis-decoded.
- **GPT-2 support.** The first family dlm supports that is not Llama-descended:
  LayerNorm rather than RMSNorm, learned position embeddings instead of RoPE, an
  ungated MLP, biases on every projection, and `Conv1D` weights stored
  transposed. CPU only for now.
- **Phi-3 / Phi-3.5 support.** Their fused `self_attn.qkv_proj`
  (`[q_dim + 2*kv_dim, hidden]`) and `mlp.gate_up_proj` (`[2*intermediate, hidden]`)
  are split at load; the block is otherwise Llama-shaped. Detected by tensor
  presence rather than by architecture string, so any checkpoint shipping fused
  projections is handled. The 128k `longrope` variant is still refused, since dlm
  does not implement that scaling and running without it yields fluent nonsense.
- **Graceful shutdown.** `SIGTERM`/`SIGINT` (Unix) stop the accept loop and drain
  in-flight requests for up to 8 seconds, under the 10s grace period Docker and
  Kubernetes allow before `SIGKILL`. Previously every deploy cut live requests
  and left SSE streams without their terminating chunk.
- Tests for the HTTP hardening constants — request-line length, header count,
  body size, the connection cap, and chunked-encoding refusal — none of which
  had been exercised.
- `cargo-deny` in CI (advisories, licenses, sources) plus a weekly schedule, and
  Dependabot for `cargo` and `github-actions`.

## [0.3.0]

Baseline for this changelog. Earlier history is in the git log; see
`RELEASING.md` for the verification ledger as it stood at 0.3.0.

### Known gaps at 0.3.0

Carried forward and tracked in `RELEASING.md`:

- Only Turing-class hardware (GTX 1650, 4 GB) had run the GPU code.
- The ROCm/HIP compute path compiled but had never executed on an AMD card.
- Multi-GPU and distributed serving were untested on real hardware.
- No context near the 8192 default had been exercised — significant for Gemma2,
  whose sliding window is only reachable past 4096 tokens.
- DeepSeek-V2/V3 (MLA + MoE) had run only on the CPU path.
- Llama 2 was expected to work but had not been demonstrated.

[Unreleased]: https://github.com/vedantnimbarte/dlm/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/vedantnimbarte/dlm/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/vedantnimbarte/dlm/releases/tag/v0.3.0
