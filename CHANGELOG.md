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

- **API compatibility: four things that turned working clients away.**
  - **Every model was called `dlm`.** The id in each response and in
    `/v1/models` is now the checkpoint's own name (`qwen2.5-0.5b`, or a GGUF
    file's). A client picking from the model list, routing between instances, or
    checking a reply against what it asked for had nothing to go on.
  - **`content` as an array of parts is accepted**, not only as a string. The
    OpenAI SDKs and anything that can attach an image send the array form, and
    dlm answered 400. Parts it cannot serve (an image) are dropped rather than
    refused, and `content: null` reads as empty.
  - **A request without `max_tokens` now runs to EOS**, instead of stopping at
    128 tokens with `finish_reason: length`. It is still clamped to what is left
    of the context window, which is the cap that matters.
  - **`--cors-origin`** answers browser preflights and tags replies, so a browser
    UI can reach the API. Without it the request succeeds and the browser
    discards the answer, with nothing in the log. Off by default: a server on a
    shared network should not hand its answers to any page that asks.
- **A K-quant sub-block with a zero scale decoded to zero on the CPU.** ggml
  decodes it as `-dmin * m`, and so do the GPU kernels; the CPU reference and
  the expanded (int4/int8) layout dropped the minimum. Both now keep it.
- **The end-to-end server test drove an engine nothing served.**
  `tests/server.rs` exercised a second, older OpenAI router that only the test
  referenced, so its checks said nothing about the server `dlm serve` runs. It
  now drives `EngineService`, and the dead router is gone.

### Changed

- **A single-token layer's attention half takes fewer launches.** Two merges
  that need no grid-wide synchronization:
  - **RoPE and the K/V append run inside the attention kernel**
    (`attn_split_post_kernel`), removing `qk_post`'s launch. Each attention
    block normalizes and rotates its own query heads in shared memory; the
    block for the last chunk, the only one that reads the token's own row,
    rotates its key and appends K and V before attending. Qwen2.5-0.5B batch
    1: Q4_K_M 132 -> 137 tok/s, bf16 97 -> 103; Qwen2.5-1.5B unchanged (65.6
    / 65.7 over six runs).
  - **For int8 activations, the input and post-attention norms and their
    quantization are one launch** (`rmsnorm_q8_kernel`), and the GEMVs that
    read the row (Q/K/V; gate and up, which also stop quantizing it twice)
    reuse its codes. Batch 1, four alternating pairs: Qwen2.5-0.5B Q8_0 129 ->
    138 tok/s, int8 129 -> 145. Q4_K_M at 0.5B is unchanged: its 896-wide
    rows do not take int8 activations.

  A single kernel for the whole half would need a grid-wide sync between the
  Q/K/V GEMV, attention and the output GEMV (a cooperative launch, which
  limits the GEMVs' occupancy); these two cover the launches that need none.

- **Single-request decode is 13-22% faster: the fused GEMV no longer spills
  to the stack.** The kernel that runs a layer's Q/K/V, output and down
  projections for one sequence found each row's matrix by indexing its
  segment struct with a runtime value, which copied all 88 bytes of it to
  every thread's stack, and then read the row's weight pointer from that stack
  inside its inner loop. It now selects with constant indices and takes the
  weights as their own read-only parameters (`STACK:0`). Batch-1 decode,
  GTX 1650, tok/s: Qwen2.5-0.5B bf16 85 -> 104, Q4_K_M 110 -> 133, Q8_0
  109 -> 130, Qwen2.5-1.5B Q4_K_M 59 -> 67. bf16 batch 1 now matches the 101
  tok/s vLLM (fp16) made on this card. Batched decode runs other kernels and
  is unchanged.

- **N-gram speculation now pays on the GPU, and `--speculative auto` turns it
  on there.** A speculative round scores the pending token and its proposals
  in one target pass, but on the GPU that pass went through the prefill GEMM,
  which computes 64-token tiles, and each position then ran the LM head on its
  own and copied its logits to the host. Verifying 5 tokens cost what decoding
  them one by one did, so speculation could not win, and `auto` left it off for
  resident models. Now a piece of 16 tokens or fewer goes through the batched
  decode block (a verify, a short prompt, a long prompt's tail), and the verify
  scores every position in one LM-head call (`Generator::distributions`), only
  the argmax coming back under plain greedy. Batch 1 on a GTX 1650, n-gram
  lookup proposing 4, decode tok/s (`dlm bench --speculative ngram`, which
  now measures it):

  | Model | Plain | Before | After | Accepted |
  |---|---|---|---|---|
  | Qwen2.5-0.5B bf16, greedy | 85 | 84 | 317 | 100% |
  | Qwen2.5-0.5B Q4_K_M, greedy | 111 | 98 | 358 | 100% |
  | Qwen2.5-1.5B Q4_K_M, greedy | 58 | 33 | 130 | 100% |
  | Qwen2.5-0.5B Q4_K_M, temperature 1.2 | 71 | | 79 | 39% |
  | Qwen2.5-1.5B Q4_K_M, temperature 1.2 | 52 | | 65 | 78% |

  The bench prompt repeats, the case lookup is for, so the greedy rows are
  its best case; real text proposes less often. A round whose every guess
  misses costs about a third of a step more than a plain step (a 5-token
  verify is 1.55x one). Several requests at once still decode plainly in one
  batched pass, as before. Short prompts gain too: time to first token for 8
  / 16 tokens 66 / 58 ms -> 30 / 39 ms. `--speculative off` restores plain
  decoding; `dlm bench` stays plain unless asked.

- **A batched decode layer launches Q/K/V as one GEMV and gate/up as
  another.** Each layer issued about 21 launches, and the host spent 5-8 ms
  of a batched step issuing them; on Qwen2.5-0.5B, K and V alone were 128 rows,
  8 blocks each on a 14-SM card. Matrices that share a dtype now go through
  one launch over their row segments, with the activations quantized once.
  GTX 1650, Qwen2.5-0.5B, decode tok/s at batch 4 / 8 / 16, the previous commit
  and this one alternately: bf16 311 / 410 / 473 -> 328 / 408 / 487, Q8_0
  435 / 654 / 760 -> 480 / 637 / 798. A GGUF layer whose V is Q6_K beside
  Q4_K Q and K keeps three launches.

- **Long-prompt prefill is up to 1.6x faster: attention is tiled over
  queries.** Prefill attention gave each prompt token its own block, which read
  the whole history, so a 512-token chunk at 8k positions read about 2 GB of
  K/V per layer. A block now takes a tile of tokens (times the query heads
  sharing a KV head) and streams the history through shared memory once for
  all of them, with a running softmax. Heads 64 and 128 wide (Qwen, Llama,
  Mistral); others keep the per-token kernel. GTX 1650, prefill tok/s, per-
  token and tiled kernels alternately:

  | Prompt | Qwen2.5-0.5B bf16 | Qwen2.5-1.5B Q4_K_M |
  |---|---|---|
  | 512 | 1,350 -> 1,490 (noise) | 486 -> 489 |
  | 2,048 | 1,218 -> 1,447 | 397 -> 435 |
  | 8,000 | 651 -> 1,018 | 234 -> 303 |

- **`--kv-quant int8` (and `int4`) now keeps the GPU KV cache in int8: 1.76x
  or more the tokens in the same VRAM.** Both used to mean fp16 on the GPU.
  Each 32 values are now a float scale and 32 int8 codes, 1.125 bytes a value
  against fp16's 2; models whose heads are not a multiple of 32 wide keep
  fp16. The pool sizing and the fit check count it, and the server prints the
  pool (`kv cache : N tokens`). On a GTX 1650, Qwen2.5-1.5B Q4_K_M at
  `--context-length 32768` with four requests: 74,608 tokens of fp16 KV fit
  beside the weights, against all 131,072 in int8. Decode and prefill speed
  are within a few percent of fp16 (0.5B Q4_K_M, 3,000-token prompt: batch 8
  decode 354 -> 365 tok/s, prefill 1,130 -> 1,090).

  **It costs accuracy, so fp16 stays the default.** Scoring 833 tokens of
  this README through the decode kernels (`dlm score --decode`) on
  Qwen2.5-0.5B: mean NLL 2.841 with fp16 KV, 2.901 with int8 (the CPU's int8
  store, the same scheme: 2.878). Qwen's keys have a few channels far larger
  than the rest, and a 32-value group holding one quantizes its neighbours
  coarsely; per-channel key scales are the known fix. The GPU's int8 agrees
  with the CPU's to 1e-7 on the same inputs.

- **Sampled decode is 4-8x faster at batch 16: candidates are found on the
  GPU.** A temperature/top-k/top-p step copied each sequence's whole row of
  logits to the host (600 KB on Qwen2.5) and found the softmax total and the
  top candidates there, one sequence at a time; at batch 16 that took several
  times as long as the model. The device head now returns each row's largest
  logit, its softmax total and its largest logits sorted (4 KB for top-k 40),
  and the host draws from those with the same arithmetic. A draw they cannot
  decide exactly still takes the logits: a top-p nucleus wider than 1,024
  tokens, pure temperature with neither top-k nor top-p, a repetition or
  frequency penalty, a grammar, or a final logit softcap. The draws match the
  host sampler's for a seed, except that the vocabulary total is summed in
  another order and can differ in its last bit. GTX 1650, Qwen2.5-0.5B Q4_K_M,
  temperature 0.8, decode tok/s:

  | Sampler | Batch 1 | Batch 4 | Batch 16 |
  |---|---|---|---|
  | top-p 0.95 | 33 -> 120 | 57 -> 340 | 83 -> 673 |
  | top-k 40 | 36 -> 115 | 106 -> 345 | 153 -> 678 |
  | greedy, for reference | 115-128 | 341-367 | 712-722 |

  `dlm bench` takes `--temperature`, `--top-p` and `--top-k` to measure it,
  and records the sampler in its JSON.

- **Batched decode of quantized models is 2-3x faster at batch 8-16.** The
  int8-activation GEMV that every quantized format (int8, int4 and the GGUF
  types) takes in a batch gave each warp one weight row and decoded each
  32-weight chunk again for every sequence. Past batch 4 that decoding, not
  memory, set its speed: about 250 tok/s on Qwen2.5-0.5B in any quantized
  format, barely above batch 4. Each format's chunk decode is now separate
  from its dot product, so a warp decodes four rows' chunks once and dots them
  with every sequence's activations. The previous commit and this one
  alternately on a GTX 1650, decode tok/s:

  | Model | Batch 1 | Batch 4 | Batch 8 | Batch 16 |
  |---|---|---|---|---|
  | 0.5B Q4_K_M | 125 -> 124 | 197 -> 335 | 211 -> 513 | 237 -> 658 |
  | 0.5B int4 | 117 -> 124 | 196 -> 413 | 217 -> 615 | 236 -> 672 |
  | 0.5B Q6_K | 95 -> 95 | 121 -> 215 | 181 -> 340 | 226 -> 500 |
  | 1.5B Q8_0 | 42 -> 41 | 72 -> 148 | 76 -> 223 | 76 -> 247 |

  The Q6_K runs drifted widely (batch 1 from 55 to 118 tok/s on either build):
  its batch 4-16 figures are medians of three, its batch 1 the one pair run
  back to back on a steady card. Q6_K's one-sequence dot keeps its old fused
  form: through the split decode it ran 5-10% slower at batch 1.

- **Batched f32/bf16/f16 projections take four weight rows per warp.** At
  batch 8-16 the batched GEMV was bound by loads, not memory: it re-read
  every sequence's activations for each weight row, eight floats for eight
  multiply-adds. A warp now decodes four rows' chunks and multiplies each
  activation chunk it loads into all four, with the batch size fixed at
  compile time (4, 8 or 16) so the accumulators stay in registers. The
  quantized formats keep their int8-activation kernel. The previous commit
  and this one alternately, twice, same setup as below:

  | Batch | Before | After |
  |---|---|---|
  | 4 | 270 | 310 |
  | 8 | 291 | 419 |
  | 16 | 324 | 496 |

  Batch 1 and prefill take other kernels and measured the same once the run
  order was reversed (the card runs warmer later in a series).

- **Batched attention runs in three launches for the whole batch.** The fused
  batched block read each weight once for all sequences, but still launched
  Q/K norm, RoPE, the K/V append and split-K attention slot by slot: 2-3 small
  launches per sequence per layer, about 1.5 ms a sequence per step on
  Qwen2.5-0.5B. Now each launch covers every slot, which reads its own cache,
  block table and history length. Shapes the split kernel cannot take still go
  slot by slot. Same setup as the LM-head entry below, three builds run
  alternately twice (the first `main` run was throttled and is left out):

  | Batch | Before both | LM head shared | Plus this |
  |---|---|---|---|
  | 4 | 166 | 217 | 249 |
  | 8 | 178 | 228 | 276 |
  | 16 | 180 | 238 | 300 |

  vLLM 0.33 in fp32 on the same card made 200 / 281 / 286 tok/s at these batch
  sizes (`bench/BASELINES.md`).

- **Batched decode is about 30% faster: the batch shares one LM-head pass.**
  Each session used to run the vocabulary-wide LM head on its own, reading
  its weights once per sequence: on Qwen2.5-0.5B at batch 16 that was 36 ms of
  a 90 ms step. `step_sessions` (the server's scheduler and `dlm bench`) now
  sends every session's hidden state through one call, which reads the weights
  once per 16 rows; greedy rows still get only their argmax back. GTX 1650,
  Qwen2.5-0.5B bf16, `--prompt-len 256 --gen-len 64`, `main` and this change
  run alternately four times, median decode tok/s (batch 1 is the same
  single-row path, run twice each, so its range is shown):

  | Batch | Before | After |
  |---|---|---|
  | 1 | 85-89 | 79-88 |
  | 4 | 153 | 203 |
  | 8 | 165 | 218 |
  | 16 | 181 | 235 |

- **GPU prefill is 4.6-6x faster and decode up to 1.7x.** Measured on a GTX
  1650 (4 GB), `dlm bench --context-length 2048 --prompt-len 512 --gen-len 128
  --batch 1,8`, the 0.4.x build and this one run alternately, best of three:

  | Model | Decode, batch 1 | Decode, batch 8 (total) | Prefill |
  |---|---|---|---|
  | Qwen2.5-0.5B Q4_K_M GGUF | 80.6 → **116.6** tok/s | 112 → **180** | 268 → **1614** |
  | Qwen2.5-0.5B bf16 | 79.3 → **89.0** | 112 → **185** | 266 → **1575** |
  | Qwen2.5-1.5B `--quant int4` | 34.1 → **59.2** | 49.8 → **64.1** | 106 → **484** |

  - **GEMV:** a warp per output row, each lane decoding an aligned chunk of
    weights with wide loads. Quantized formats dot int8-quantized activations
    with `dp4a`, as llama.cpp does: their GEMVs had been bound by decoding at
    16-40 GB/s, not by memory. Rounding the activations to 8 bits costs at
    most 0.003 nats of NLL on any weight type measured (see the table below);
    `DLM_GEMV_EXACT=1` keeps the f32 path.
  - **Fewer launches:** Q, K and V in one grid; the output and down
    projections add straight into the residual; one warp per head does
    Qwen3's head norm, RoPE and the KV append (six launches before). Fusing
    gate, up and SwiGLU was tried and dropped: it never beat three launches.
  - **Attention** splits the cache into chunks, reads each key once per KV head
    instead of once per query head, and merges long contexts' chunks in a
    second launch.
  - **Prefill** takes up to 512 prompt tokens through a layer at once as tiled
    GEMMs, with one attention launch for every query, instead of the decode
    block 16 tokens at a time. Cards with tensor cores multiply on them (bf16
    from Ampere, fp16 with per-row scaling on Volta/Turing RTX); GTX cards use
    the f32 kernel. `DLM_GEMM=simt|tc` overrides the choice.
  - **Greedy decoding** takes the LM head's argmax on the device, copying four
    bytes back instead of the vocabulary's logits.
  - **Batched decode of MoE models** reads each attention matrix once per
    batch instead of once per sequence.
  - `build.rs` builds the kernels for sm_75, 80, 86 and 89 plus PTX, so Ampere
    and newer run their own code (bf16 tensor cores need it).
    `DLM_CUDA_ARCHS=75` narrows a local build.
- **Every weight type, measured.** The same GTX 1650, `--context-length 1024
  --prompt-len 256 --gen-len 128`, the 0.4.x build against this one, run
  alternately, best of two. Decode tok/s (batch 8 is the total over the batch),
  and mean NLL over ~1,300 tokens of text scored the way generation decodes
  (`dlm score --decode`), with the int8-activation GEMVs and with
  `DLM_GEMV_EXACT=1` (noise is about 0.002):

  | Model | Decode, batch 1 | Decode, batch 4 | Decode, batch 8 | Prefill | NLL (exact → dp4a) |
  |---|---|---|---|---|---|
  | 0.5B bf16 | 96.9 → **107.9** | 108.5 → **199.2** | 138.6 → **204.4** | 290 → **1727** | 2.7265 (no change) |
  | 0.5B `--quant int8` | 89.6 → **137.7** | 100.2 → **197.5** | 130.2 → **205.4** | 293 → **1725** | 2.7298 → 2.7285 |
  | 0.5B Q8_0 GGUF | 88.2 → **135.5** | 99.9 → **200.3** | 131.6 → **208.8** | 292 → **1704** | 2.7516 → 2.7531 |
  | 0.5B Q6_K GGUF | 88.6 → **132.6** | 100.6 → **170.4** | 131.5 → **197.7** | 287 → **1732** | 2.7508 → 2.7507 |
  | 0.5B Q5_K_M GGUF | 88.5 → **133.4** | 100.9 → **183.9** | 131.9 → **202.8** | 302 → **1696** | 2.7537 → 2.7561 |
  | 0.5B Q4_K_M GGUF | 88.6 → **137.3** | 100.8 → **184.8** | 131.8 → **203.7** | 298 → **1720** | 2.7769 → 2.7793 |
  | 0.5B `--quant int4` | 90.2 → **183.1** | 101.0 → **210.4** | 132.9 → **220.4** | 308 → **1785** | 2.8669 → 2.8676 |
  | 1.5B `--quant int8` | 37.5 → **58.4** | 38.9 → **75.9** | 56.1 → **77.1** | 116 → **504** | 2.3285 → 2.3305 |
  | 1.5B Q8_0 GGUF | 36.7 → **55.7** | 38.9 → **75.5** | 55.9 → **76.9** | 116 → **499** | 2.3408 → 2.3428 |
  | 1.5B Q4_K_M GGUF | 36.2 → **72.8** | 39.1 → **72.3** | 56.7 → **79.4** | 117 → **504** | 2.3773 → 2.3764 |
  | 1.5B `--quant int4` | 38.0 → **86.3** | 39.5 → **82.8** | 58.0 → **84.2** | 119 → **531** | 2.4183 → 2.4211 |
  | 1.5B bf16, `--stream` | 18.8 → 18.3 | 33.6 → **50.8** | 49.3 → **64.6** | 116 → **124** | |

  The batch-4/8 figures include the batched int8-activation GEMV below. A
  streamed model is bound by PCIe at batch 1, so it moves little there.
- **A batch of quantized sequences decodes through the int8-activation GEMV**
  too. It used to decode weights to f32 element by element, so on Qwen2.5-1.5B
  (int8, Q8_0, Q4_K_M, int4) four sequences together made fewer tokens a second
  than one alone. Now batch 4 runs 72-83 tok/s, against 44 on the same build
  without this kernel. At batch 4-8, int4 and Q4_K_M still only about match one
  stream: each slot re-decodes the weights, which is compute-bound there.
- **`--quant int4` decodes 20-27% faster on the GPU.** On short rows a
  quantized GEMV was bound by latency, not bandwidth: a 32-weight chunk is a few
  bytes, so a warp issued one load per row, waited, and reduced with nothing in
  flight (int4 reached 60-65% of the bandwidth bf16 does, at full occupancy).
  Each warp now interleaves two rows (int4, Q4_K) or four (Q4_0-Q8_0, Q6_K),
  in the plain and the fused Q/K/V GEMVs. Every row sums in the same order as
  before, so outputs are bit-identical. GTX 1650, batch-1 decode:

  | Model | Before | After |
  |---|---|---|
  | Qwen2.5-1.5B `--quant int4` | 66.9 tok/s | **85.1** |
  | Qwen2.5-0.5B `--quant int4` | 151.6 | **182.2** |
  | Qwen2.5-0.5B Q4_K_M GGUF | 135.0 | 136.9 |

  The GGUF barely moves because a model that fits is expanded to dlm's int8,
  which was already at bandwidth; GGUF blocks read natively (a model too large
  to expand) go 1.3-1.75x faster per GEMV.
- **The quantized KV cache keeps a scale per 32 values, not per token.** One
  scale across a token's whole K or V vector let a few large key channels set
  the range for every head. Qwen2.5-0.5B, mean NLL over 415 tokens (3.644 with
  an exact cache): int8 3.750 → 3.685, int4 6.266 → 5.312. int4 KV is still
  far from exact; prefer int8. `dlm score --kv-quant` measures it.
- **CPU decode is 2.5x faster on a bf16 model and 4.5x on a GGUF one.** The
  CPU matvec summed one weight at a time (an f32 `sum()` that LLVM may not
  reorder) and decoded every quantized weight on its own. Measured on a Ryzen 5
  5600GT with Qwen2.5-0.5B, `dlm bench --device cpu --prompt-len 128
  --gen-len 16`, old and new binaries run alternately, two rounds of three runs
  each (medians per round; the machine was shared with other benchmarks, which
  the spread shows):

  | Model | Workload | Before | After |
  |---|---|---|---|
  | bf16 safetensors | Decode | 3.8–3.9 tok/s (runs 3.7–4.4) | **9.3–11.0 tok/s** (runs 4.4–11.5) |
  | bf16 safetensors | Prefill | 4.9 tok/s | **6.3–12.9 tok/s** |
  | Q4_K_M GGUF | Decode | 1.7–2.1 tok/s (runs 1.3–2.2) | **8.9–9.1 tok/s** (runs 8.5–10.0) |
  | Q4_K_M GGUF | Prefill | 2.0–2.2 tok/s | **16.1–17.8 tok/s** |

  - **SIMD kernels for every weight dtype**: f32, bf16, f16, dlm int4/int8, and
    ggml's Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and Q6_K. AVX2+FMA+F16C on
    x86_64, picked at run time so the same binary still runs the scalar code on
    a CPU without them; NEON on aarch64. Each format's decoding is written once
    against a small set of vector operations that both instruction sets
    implement. The scalar code stays as the reference, and a test checks every
    dtype against it on random data. Activations stay f32: llama.cpp's int8
    activations would be faster for 4-bit weights, but this path is the oracle
    the GPU kernels are checked against at 2e-3, and an int8 activation's error
    is a few times that.
  - **The thread pool no longer waits for its slowest worker to wake.** Each
    GEMV dealt one fixed slice per thread and waited for every thread, so on a
    loaded machine one descheduled worker stalled the whole step. Rows now go
    out in small chunks claimed by whichever threads are running, and a worker
    that wakes after the last chunk is taken skips the call. On the shared
    machine above, a 4864×896 bf16 GEMV went from 1.7 ms to 0.18 ms (median).
  - The attention projections (896×896) now run on the pool too, and attention
    dots each query head with its keys eight lanes at a time.
  - **The CPU's LM head is held in bf16 when that is lossless** (any bf16
    checkpoint): the vocabulary-wide GEMV reads half the bytes, and steady-state
    memory drops by 272 MB on Qwen2.5-0.5B. The load-time peak RSS is ~0.2 GB
    higher (2.86 vs 2.67 GiB), since the bf16 copy is made from the f32 one the
    loader hands over.
  - Decode is now bound by memory bandwidth, not arithmetic. A GGUF model's LM
    head is still decoded to f32 on the CPU, and is over half of each token's
    bytes there; keeping it in its ggml blocks is the next step.

- **`--quant int4` chooses each group's scale and zero to fit the group, not
  just to span it.** Spanning min..max gives the two extreme weights exact
  codes and rounds the other 126 on a coarser grid than they need. Each group of
  128 now tries 21 grids around min..max and solves the least-squares scale and
  zero for each one's codes (llama.cpp's `make_qkx2_quants`, the search behind
  its K-quants, without calibration data), keeping the best. The min..max fit is
  one of the candidates, so no group gets worse; synthetic groups lose 12–15% of
  their squared error, and 35% when a few weights are outliers. The storage
  layout and kernels are unchanged. `dlm score --device cpu` on a 404-token
  text, mean NLL in nats (bf16 is the floor):

  | Model | bf16 | int4 before | int4 after | int8 | int8 with the search |
  |---|---|---|---|---|---|
  | Qwen2.5-0.5B | 2.8421 | 3.0509 | **2.9932** | 2.8426 | 2.8434 |
  | Qwen2.5-1.5B | 2.2849 | 2.4217 | **2.3847** | 2.2861 | 2.2869 |

  That is 27–28% of int4's loss gone. int8 keeps plain rounding: it is already
  within noise of bf16, and the search changed nothing measurable but doubled
  its load time. The groups are fitted on every core, so loading costs about
  2–3 s more on 0.5B and 10 s more on 1.5B (median of three, 6-core CPU); int8
  loads no slower, and on 1.5B a little faster.
- **`--quant int4` keeps `v_proj` and `down_proj` at int8 in the first and last
  eighth of the layers**, the core of llama.cpp's Q4_K_M rule: those two write
  straight into the residual stream, and the ends of the network are where it
  is least forgiving. The embedding and LM head were never int4 (f32 on the CPU,
  int8 on the GPU), so they needed nothing. Same text and setup as above, on
  top of the fitted groups:

  | Model | int4, all 4-bit | ends at int8 (kept) | + every 3rd layer (Q4_K_M) | all `v`/`down` int8 |
  |---|---|---|---|---|
  | Qwen2.5-0.5B, NLL | 2.9932 | **2.9689** | 2.9448 | 2.9121 |
  | layer weights | 192.0 MiB | 204.8 MiB | 217.5 MiB | 243.1 MiB |
  | Qwen2.5-1.5B, NLL | 2.3847 | **2.3564** | 2.3576 | |
  | layer weights | 702.8 MiB | 756.8 MiB | 797.3 MiB | |

  On 0.5B each extra int8 byte buys about the same, wherever it goes; on 1.5B
  the middle layers bought nothing. So dlm takes the ends only: 7% larger, and
  with the fitted groups, int4's loss against bf16 falls from 0.209 to 0.127
  nats on 0.5B and from 0.137 to 0.072 on 1.5B. MLA layers (DeepSeek) are left
  alone, since the GPU runs their dense FFN at a single dtype.
- **The GPU's batched GEMV read every weight once per sequence instead of once
  per batch, and both GEMVs spent a third of their time in barriers.** Fixing
  the two, measured on Qwen2.5-0.5B by running the old and new binaries
  alternately on the same machine:

  | Workload | Before | After |
  |---|---|---|
  | Prefill (resident) | 116–152 tok/s | **251–268 tok/s** |
  | Prefill (streamed int4, 1.5B) | 40–43 tok/s | **95–106 tok/s** |
  | Decode, batch 8 | 107–109 tok/s | **131–136 tok/s** |
  | Decode, batch 1 | 74–83 tok/s | **79–87 tok/s** |

  - **One read per batch.** `matvec_batched_kernel`'s comment claimed it held a
    row across the batch loop; the code re-read the whole row from global memory
    for every slot. It now keeps one accumulator per slot in registers and
    multiplies each weight element into all of them as it is read. Prefill feeds
    a chunk of tokens through this kernel, which is why it gains most.
  - **Warp-shuffle reductions.** Both GEMVs reduced through shared memory, one
    barrier per tree level — and the batched one did that per slot, so a 896-wide
    projection cost 48 barrier rounds against 14 rows of work. They now reduce
    inside each warp by shuffle and combine the warps once.
  - Summation order within a row is unchanged; the order the partial sums combine
    in is not, which moves results within float rounding. Verified against
    Hugging Face transformers after the change: Qwen2.5-0.5B, Gemma 3 1B and a
    Qwen2-MoE all still agree to 0.0001, and the 46 GPU parity tests pass.
  - MoE gains little (~2%): its expert FFN runs per token through the
    single-slot GEMV, and MoE decode does not use the batched block at all.

### Added

- **2- and 3-bit GGUF files: Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ4_NL and IQ4_XS.** These were refused by name, which ruled out every
  `Q2_K`, `Q3_K_*`, `IQ2_*`, `IQ3_*` and `IQ4_*` download, the files that make a
  model fit a small card. They decode on the CPU and the GPU, in the GPU's
  int8-activation GEMV, its f32 chunk path and its prefill GEMM, from one
  decoder per format on each side; the IQ codebooks are ggml's (MIT), generated
  from `ggml-common.h`. IQ1 is still refused.
  - **Checked against llama.cpp**: `tests/ggml_blocks.rs` decodes random blocks of
    every type against llama.cpp's reference decoder (a 55 KB fixture, so it runs
    in CI), and Qwen2.5-0.5B `Q2_K` and `Q3_K_M` decode to llama.cpp's f32 copy
    of every tensor. On the GPU, the identity matrix through the LM head returns
    each decoded weight exactly as the CPU decodes it (`tests/gpu_parity.rs`).
  - **Quality**, Qwen2.5-0.5B on one 228-token passage, nats per token (GPU; the
    CPU agrees within 0.006): `Q4_K_M` 3.31, `IQ3_XS` 3.33, `IQ3_M` 3.35,
    `Q3_K_M` 3.36, `Q2_K` 3.38, `IQ2_M` 3.57. Layer bytes 235, 179, 183, 195,
    179 and 169 MiB.
  - **Speed**, GTX 1650, Qwen2.5-0.5B, decode tok/s at batch 1 / 16: native
    blocks `Q2_K` 107 / 616, `IQ3_XS` 99 / 608, `IQ2_M` 111 / 594. When the
    model fits with VRAM to spare, the loader re-packs them into dlm's int
    layout as it does for the other GGUF types, which is faster resident
    (`Q2_K` 140-147 / 584-604, `IQ3_XS` 144-146 / 787-792) and three to four
    times the bytes; a model that would not fit keeps the blocks. Q2_K and
    Q3_K re-pack to int8, not int4: their 16-weight groups miss the GPU's
    32-wide int4 chunks, which halved a `Q2_K` file's prefill (about 700
    against 1,480-1,520 tok/s).

- **`--lm-head-int4`: hold the GPU's LM head in int4.** The vocabulary-wide
  head is a large share of a small model's bytes per token (272 MB of
  Qwen2.5-0.5B bf16's ~1 GB). Opt-in, since it costs accuracy; without it the
  head stays as before (the checkpoint's precision, or int8 for `--quant`).
  `place_lm_head_on_gpu` takes an `LmHeadQuant` (`Native`, `Int8`, `Int4`)
  in place of its `bool`. GTX 1650, Qwen2.5-0.5B bf16, two alternating pairs:
  batch-8 decode 404 / 371 -> 441 / 406 tok/s (+9%), batch 1 109 / 102 ->
  115 / 99 (noise). Mean NLL over 833 tokens (`dlm score --decode`): 0.5B
  bf16 2.841 -> 2.865, 0.5B Q4_K_M 2.908 -> 2.928, 1.5B Q4_K_M 2.525 ->
  2.567. It pays most on a bf16 head; a GGUF head is already 6-8 bits.

- **`dlm score --decode`** scores a text one token at a time, through the
  kernels generation uses. On the GPU those differ from prefill's (GEMVs on
  int8-quantized activations against GEMMs on f32 ones), so plain `dlm score`
  does not measure what a generated token goes through.
- **`--gpu-layers N`: the layers that do not fit in VRAM run on the CPU**,
  where their weights already are, instead of streaming across PCIe every
  token (llama.cpp's `-ngl`). Qwen2.5-1.5B bf16 on a 4 GB GTX 1650, against
  `--stream` holding the same number of layers: decode 8.3 → 11.0 tok/s at 16
  of 28 layers, 5.6 → 7.4 at 8. Prefill is slower (the CPU runs a prompt token
  by token), so `--stream` remains the better choice for long prompts.
- **Speculative decoding without a draft model, on by default where it pays.**
  N-gram lookup proposes what followed the last time the context's latest few
  tokens occurred; the target verifies the proposals in one pass, and keeps
  the ones it agrees with. It is right whenever a reply copies — code being
  rewritten, a document being quoted, a list being continued — and costs
  nothing to draft.
  - **`--speculative auto|off|ngram|draft`** on `serve`. `auto`, the default,
    uses the draft model if `--draft-model-path` is given, else lookup when the
    model is streamed, else nothing. Streamed means a pass moves weights: layers
    left out of the `--stream` window, and on the CPU, more of them than the RAM
    cache holds. That is where verifying several tokens costs about what
    generating one does; where a pass is compute, a wrong guess is paid for in
    full. `--draft-gamma` sets the proposals per round for either drafter.
  - **Output is unchanged.** Greedy is exactly target-greedy (tested against
    plain decoding, alone and in the batching scheduler). A lookup proposal is
    a one-hot draft, so the usual rejection rule keeps it with probability
    `p(x)` and otherwise resamples from `p` without it: sampled output follows
    the model's own distribution (tested on the rule directly). Penalties and
    `logit_bias` hold too: the target's distribution gets them at each
    proposal's own position, and a one-hot proposal needs none. A constrained
    request decodes plainly, as with a draft model.
  - **Acceptance is reported** in `usage.speculative`, as for a draft model.
  - **Concurrency keeps its shared pass.** A request speculates only while it
    is the one decoding; with several, each takes one token through the same
    batched pass as plain decoding, since the streamed GPU kernel streams each
    layer once for the whole batch and a verify pass per request would undo
    that. A round with no match is a plain step, not a one-token verify.
  - **Prompts still go in piecewise and through the prefix cache**: a lookup
    request starts as a plain one and turns speculative once its prompt is in,
    at no extra pass.
  - **Measured** on Qwen2.5-0.5B, CPU build, `--stream --resident-layers 12`,
    greedy, 96 tokens, decode only, on a machine shared with other work (the
    quieter round):

    | | Rewrite code with comments | Free prose |
    |---|---|---|
    | Load-bound (`--ram-cache-gb 0`), off | 3.5 tok/s | 3.7 tok/s |
    | Load-bound, lookup | **6.3 tok/s** (79% of 87 accepted) | 3.6 tok/s (17% of 24) |
    | Layers all in the RAM cache, off | 8.3 tok/s | 8.6 tok/s |
    | Layers all in the RAM cache, lookup | 7.6 tok/s | 7.6 tok/s |

    The last two rows are why `auto` leaves a CPU model whose layers all fit
    the RAM cache alone: the CPU kernels verify token by token, so a pass over
    five tokens costs nearly five passes. Not measured: the streamed GPU path,
    the case this is for, since the GPU was busy.
  - **Lookup needs a two-token match.** One recurring token predicts little:
    on the prose above, 1-token matches made 129 proposals of which 6% were
    accepted, and cost ~10% of decode speed even load-bound. Two tokens cut that
    to 24 proposals with no loss on the code rewrite (70 of 90 accepted before,
    69 of 87 after). Adapting the proposal length to recent acceptance was tried
    and dropped: it shortened the long runs a rewrite copies (2.7–3.0 tok/s
    against 4.1–4.6) and helped prose less than the two-token minimum did.

- **`logit_bias`** on `/v1/chat/completions` and `/v1/completions`, in OpenAI's
  shape: token id → a bias in [-100, 100] added to that token's logit. It
  applies under greedy decoding too, so `-100` bans a token whatever the
  temperature. A bias out of range, or a key that is not a token in the model's
  vocabulary, is a 400 that says which.
- **`frequency_penalty` and `presence_penalty`**, with OpenAI's meaning: each
  time a token has been generated takes `frequency_penalty` off its logit, and
  having been generated at all takes `presence_penalty` once. Only generated
  tokens count, not the prompt's. Range -2 to 2; negative values encourage
  repetition. Speculative decoding applies them at each proposal's own
  position, so its output still follows the target exactly.
- **DRY ("don't repeat yourself") sampling**, with llama.cpp's fields:
  `dry_multiplier` (0, the default, leaves it off), `dry_base` (1.75),
  `dry_allowed_length` (2) and `dry_sequence_breakers` (`["\n", ":", "\"",
  "*"]`). It penalizes the token that would extend a run already seen earlier
  in the context, by `multiplier · base^(run − allowed_length)` — the verbatim
  loops a small model falls into, which per-token penalties miss because every
  token in them is ordinary on its own. A run does not cross a token containing
  a breaker, so repeating a structure line by line is not counted as a loop.
- **Constrained output: JSON mode, JSON Schema, and GBNF grammars.**
  `response_format: {"type": "json_object"}` returns a JSON object;
  `{"type": "json_schema", "json_schema": {"schema": ...}}` returns JSON of
  that shape; and `grammar` takes a GBNF grammar, llama.cpp's format, so
  grammars written for it work unchanged. The output matches every time, not
  usually: at each step a token the grammar cannot take is never emitted, and
  EOS only once the grammar is complete.
  - **How:** the sampler draws as it would anyway, and only when the grammar
    refuses that token does it drop every refused candidate and draw again from
    what is left. The grammar is matched against each token's bytes, so a
    character split across two tokens is handled, and tokenization and grammar
    never need to line up.
  - **Schemas:** `type` (object, array, string, number, integer, boolean, null,
    or a list of them), `properties` with `required` (no other keys), `items`,
    `enum`, `const`, `anyOf` and `oneOf`. Keywords that would otherwise be
    silently ignored — `$ref`, `allOf`, `not`, `if`, `patternProperties` — are
    refused, as are grammars that do not parse, with a 400 naming the problem.
    Properties come out required first, each group in key order; numbers are
    capped at 16 digits and whitespace runs at 21 characters, so a model that
    favours digits or spaces still has to finish.
  - **Speculative decoding is skipped** for a constrained request, which
    decodes plainly beside the speculative ones: the rejection rule would need
    the draft held to the same grammar.
- **`dlm bench --compare base.json`** checks a build for a speed regression:
  it prints each metric's change against a baseline saved with `--json`, per
  batch size, and exits non-zero if prefill or decode tok/s fell by more than
  `--tolerance` percent (default 5). Latency and memory are shown but never fail
  it. The `--json` file now also records the time, the GPU's name, per-batch
  peak RSS, and the commit when the build sets `DLM_GIT_COMMIT`; files written
  before this still read. `bench/BASELINES.md` describes the workflow.
- **`dlm score --quant int4|int8`** scores a model quantized at load, so what a
  quantization costs in log-likelihood can be measured rather than guessed. It
  also reports the size of the layer weights it scored.
- **dlm reads GGUF files** — the format local models are actually distributed
  in. `--model-path model.gguf` runs one on CPU or GPU, resident or streamed,
  and the file is self-contained: dlm takes the config and the tokenizer out of
  its metadata, so nothing else needs downloading.
  - **Quantization:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K
    and Q6_K. Each is decoded into dlm's own group-affine form, which every
    existing kernel already reads, so this needed no new kernels. Anything else
    (Q2_K, Q3_K, the IQ types) is refused by name, with the message saying a
    Q4_K_M or Q5_K_M download is the usual choice.
  - **Verified against llama.cpp on the same file**, which is the only reference
    that can catch a misread scale: every quantized tensor of a Qwen2.5-0.5B
    Q4_K_M decodes to within 4.8e-8 of llama.cpp's own dequantization, and a
    Q8_0 one to the bit. Greedy output matches token for token except where the
    top two candidates sit closer together than the two runtimes' arithmetic
    differs — llama.cpp quantizes activations to int8 for its dot products,
    where dlm dequantizes the weights and computes in f32.
    `tools/llamacpp_parity.py` runs the comparison and reports which of the two
    a divergence is.
  - **Tokenizer:** the vocabulary, merges and token types come out of the
    metadata, byte-level (GPT-2, Llama 3, Qwen) or SentencePiece (Llama 2,
    Mistral, Gemma). Tokenization matches llama.cpp on prose, code, accented
    Latin and CJK.
  - **`--quant` is refused** for a GGUF file rather than ignored: its weights are
    already quantized, and there are no floats left to quantize.
  - **The kernels read ggml's blocks directly**, so a streamed or tight-on-VRAM
    model costs exactly what the file does. Where there is room to spare, the
    loader re-packs the blocks into dlm's own layout instead, which the kernels
    read faster. Measured on a Qwen2.5-0.5B Q4_K_M:

    | | ggml's blocks | dlm's layout |
    |---|---|---|
    | Weights per layer | **10.4 MiB** | 18.8 MiB |
    | Decode, resident on GPU | 54 tok/s | **82 tok/s** |
    | Decode, streamed through a 0.6 GiB window | **19 tok/s** | 16 tok/s |
    | Decode, CPU | 3.6 tok/s | **4.2 tok/s** |

    So the resident paths expand and the streaming paths do not, which is the
    faster choice in each case. The two are exactly equal weight for weight —
    same codes, same arithmetic — and a test pins that, as does a second one
    that generates the same tokens both ways from the same file.
  - **`dlm search --gguf`** lists repos that ship GGUF, and **`dlm pull <repo>
    --file q4_k_m`** fetches one file out of one. A GGUF repo holds the same
    model at a dozen quantizations, so the pull names the file — by its name or
    any part of it — and listing what a repo has is what leaving `--file` off
    does. `dlm pull` on a GGUF repo used to fail with "dlm cannot load
    GGUF/PyTorch-only repos".
  - **Not yet:** MoE files, whose experts are stacked into one tensor per layer.
    Q2_K and Q3_K files usually store their embedding as an IQ type, so they are
    refused as well.

- **A GGUF layer mixes quantization types, and the GPU block read them all as
  one.** A Q4_K_M file stores attention as Q4_K and parts of the FFN as Q6_K
  inside the same layer, while the device block took a single dtype for all
  seven projections — decoding six of them as the wrong format. Each projection
  now carries its own dtype and group size. Safetensors checkpoints are
  unaffected (a layer there is one dtype), and the 46 GPU parity tests confirm
  it.

- **Mixture-of-Experts models run on the resident GPU path.** Before, `--device
  gpu` refused any MoE checkpoint — `serve`, `bench` and `score` all failed with
  "use the streaming GPU kernel (serve --stream)". A MoE model whose experts fit
  in VRAM now holds them all there, with no per-expert cache in the way.
  - Measured on a tiny Qwen2-MoE: **+6% decode** (226 vs 213 tok/s) and **+27%
    prefill** over the same model streamed.
  - The routing itself is now written once (`GpuMoeCore::run`) and shared by both
    GPU kernels; only where the experts come from differs. The device kernels
    were already there — this was the missing buffers and branch.
  - MLA + MoE (DeepSeek-V2) is resident-capable too; its refusal is gone.
  - Verified against Hugging Face transformers on a real MoE checkpoint: exact
    agreement (0.0000 log-probability difference) on CPU, resident GPU, and
    streamed GPU. The GPU parity tests now hold both GPU paths to the CPU oracle.
  - Batched decode still runs MoE one sequence at a time on both GPU paths; the
    device's batched block runs a single dense FFN for the whole batch, and
    routed experts are per token.

- **`dlm score` and `tools/hf_parity.py`: a check against transformers.**
  `dlm score` reports the log-probability the model gives each token of a text
  and the distribution it would generate from next, as a table or JSON.
  `tools/hf_parity.py` runs the same text through Hugging Face transformers and
  compares.
  - **Why:** every existing test compares dlm against itself — CPU against GPU,
    streamed against resident, run against run — and all of them pass while the
    whole model is wrong in the same way. Log-probabilities, not generated text:
    greedy output stays readable long after the probabilities have drifted.
  - **Verified** to within 0.0001 log-probability of a float32 transformers
    reference: Qwen2.5-0.5B (CPU and GPU), GPT-2, Gemma 3 1B.
  - Keep the reference in float32. A bfloat16 one rounds more than dlm does:
    the same Qwen2.5-0.5B that matches float32 to 0.0001 differs from bfloat16
    by up to 0.49.

### Fixed

- **Byte-level tokenizers split text differently from the models' own
  tokenizers.**
  - **Cause:** dlm's pre-tokenizer only split before spaces. GPT-2, Llama 3 and
    Qwen split with a regex: whitespace runs become their own token, line breaks
    end a piece, contractions and digits split off.
  - **Effect:** prompts with code indentation, repeated spaces, newlines or
    digits reached the model as token sequences it was never trained on. On 411
    mixed strings, dlm matched Hugging Face `tokenizers` for only 58% (Qwen2.5),
    54% (Llama 3) and 71% (GPT-2) of them.
  - **Fix:** the pre-tokenizer implements each family's regex exactly (Qwen2 and
    Qwen3, Llama 3, GPT-2), chosen from `tokenizer.json`. `\p{L}` and `\p{N}`
    use generated Unicode 16 category tables, not `is_alphabetic`, which counts
    Indic combining marks as letters.
  - **Result:** all 411 strings now match for Qwen2.5, Qwen3, Llama 3 and GPT-2.
  - **Not covered yet:** Falcon and DeepSeek chain splits of their own and get
    the GPT-2 rule, which is much closer than splitting at spaces but not exact.
    Unicode NFC normalization (Qwen's normalizer) is not applied.

- **Config keys that change the model's behavior were read past in silence.**
  Each one now takes effect, or is refused by name when dlm does not implement
  it — a refusal is the only safe answer, since routing or positioning a model
  wrongly reads as fluent, wrong output rather than an error.
  - `use_sliding_window: false` (every Qwen2/2.5 config, next to a 32k
    `sliding_window`) now means full attention. dlm had been windowing
    attention the model expects whole, visible only past 32k tokens.
  - `tie_word_embeddings: true` now reads the LM head from the embedding matrix
    even when the checkpoint also ships an `lm_head.weight`, as transformers
    does. Only a *missing* `lm_head.weight` used to tie them.
  - Refused: `partial_rotary_factor` below 1.0 (Phi, GPT-NeoX, GLM rotate part
    of each head; dlm rotates all of it), `max_window_layers` when it windows
    only part of the model, and DeepSeek-V3's router (`scoring_func: sigmoid`,
    `topk_method: noaux_tc`, `n_group`/`topk_group`, `routed_scaling_factor`).
    DeepSeek-V2, whose routing dlm does implement, still loads.
- **`dlm pull` downloaded Mistral weights twice.** Those repos ship the same
  tensors as sharded files with an index and again as
  `consolidated.safetensors`. dlm loads the sharded copy, so the second one is
  now skipped — for multi-gigabyte repos it had doubled the download, the disk
  use, and the address space mapped at load.
- **Streamed GPU decode could read weights while they were being overwritten.**
  A layer evicted from the VRAM window could get the next layer uploaded into
  its buffers while kernels launched on the evicted layer were still queued.
  Output was then wrong, and different from run to run. Uploads now wait for
  the default stream first.
- **`GpuKernel::run_block_batched` corrupted all but the last sequence** when it
  fell back to one sequence at a time (more than 16 slots, or MLA).
  `generate_batch` reached that path.
- **One failed scheduler tick left the server spinning.** The engine closed every
  client's stream but kept their requests in the scheduler, which failed again
  on every tick after. It now aborts them as well.
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

- **Long prompts no longer stall other requests' streams.**
  - **Before:** a request's prompt was prefilled whole when it was admitted,
    and every running stream waited for it.
  - **Now:** admitted prompts are fed in 16 tokens at a time, and a tick stops
    after 200 ms while other requests are decoding. With nothing decoding,
    prompts still go in whole.
  - **Identical output:** the GPU kernels prefill in groups of 16, so pieces of
    that size leave the output bit-identical.
  - **Prefix sharing in bursts:** a request that shares at least a KV block of
    its opening with a prompt still being prefilled waits for that prompt to be
    cached, so a burst with one system prompt computes it once.
  - **Measured, GTX 1650, Qwen2.5-0.5B:** a stream decoding while a 1,590-token
    request arrived used to pause for 13.9 s. Now its longest gap is 0.41 s. The
    long request itself takes 17.7 s instead of 13.9 s.

- **Streamed batches load each layer once, not once per request.** The
  streaming GPU kernel now runs every sequence of a dense model through a
  layer's weights in one fused block call before fetching the next layer. It
  used to run each sequence through the whole stack in turn, streaming every
  non-resident layer once per sequence, so batch 4 decoded slower in total than
  batch 1.
  - **Measured, GTX 1650:** Qwen2.5-1.5B bf16 at a 4k context with 17 of 28
    layers resident. Total decode at batch 4 went from 6.7 to 24.5 tok/s, and at
    batch 8 from 6.9 to 34.9 tok/s.
  - **Unchanged:** MoE and MLA models keep the one-sequence-at-a-time path.

- **Streamed dense layers upload straight from pinned memory.**
  - **Before:** each upload first copied the layer's weights into a pinned
    staging buffer, which was the larger half of a streamed step.
  - **Now:** the GPU streaming kernel keeps dense layers staged in pinned buffers
    (an LRU, budgeted by the RAM cache size, which it replaces for these models),
    so a hit copies straight to the device.
  - **Measured, GTX 1650:** Qwen2.5-1.5B bf16 with 14 of 28 layers resident went
    from 3.2-3.5 to 7.5-7.6 tok/s. Gemma 3 4B int4 with 9 of 34 resident went
    from 2.8 to 5.6 tok/s on a quiet machine. Peak RSS is unchanged.
  - **Unchanged:** MoE and MLA layers keep the RAM cache and staging buffer.
- **The default VRAM safety margin scales with the card.** It is a tenth of
  total VRAM, between 256 MiB and 1.5 GiB. It was 1.5 GiB on every card, which
  held back nearly half of a 4 GB card's free VRAM for transient allocations
  that take tens of MiB.
  - **Qwen2.5-1.5B bf16, streamed:** 14 to 27 of 28 layers resident, and decode
    from 7.5 to 27 tok/s together with the pinned cache.
  - **Gemma 3 4B int4:** now runs resident at a 2k context without
    `--safety-margin-gb`, at 13.2 tok/s.
  - **Unchanged:** cards of 16 GB and up keep 1.5 GiB, and `--safety-margin-gb`
    still overrides it.

- **GPU decode and prefill are 1.3-1.9x faster.** All on a GTX 1650, prompt
  512, 64 generated, from four kernel changes:

  | Model | Decode tok/s | Prefill tok/s |
  |---|---|---|
  | Qwen2.5-0.5B (bf16) | 38.8 -> 49.3 | 77.7 -> 104.4 |
  | Qwen2.5-0.5B int4 | 35.6 -> 79.2 | 70.1 -> 130.7 |
  | Qwen2.5-1.5B int4 | 14.4 -> 33.5 | 22.4 -> 42.9 |
  | Gemma 3 4B int4 | 6.8 -> 14.8 | 10.2 -> 19.1 |

  At a 2k-token context, Qwen2.5-0.5B int4 decode went from 36.0 to 56.4 tok/s.
  - **GEMV reduction:** 64 threads per block instead of 256. Every doubling
    adds a barrier to each row's reduction, and 64 won on every model and width
    measured.
  - **Quantized GEMV:** scales and zeros are found once per row instead of once
    per weight, which removes three integer divisions per weight element.
    Results are bit-identical.
  - **Attention softmax:** runs across each head's row instead of walking it
    serially. At 2k context that took one token's softmax from 10.8 ms to
    0.7 ms.
  - **Device lookup:** each kernel call asks for the current device once,
    instead of about 30 times per layer.
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

- **Requests that repeat an earlier prompt's opening reuse its KV on the GPU.**
  This covers a shared system prompt, or a chat whose history grows each turn.
  - **How:** the prefix cache holds prompts cut back to whole KV blocks, and a
    snapshot shares those blocks with the session that computed them (the pool
    counts each block's holders). It used to copy the prompt's whole KV back to
    the host on every request and upload it again on a hit.
  - **Matching:** a request reuses the longest run of leading tokens it shares
    with any cached prompt, not only a cached prompt it fully extends. Requests
    with one system prompt and different questions therefore hit.
  - **Measured, GTX 1650:** Qwen2.5-0.5B with a 1,590-token system prompt. The
    first request took 14.1 s. Later requests with different questions took
    0.27 s, and their answers were identical to the same question computed
    fresh.
  - **Default:** on the GPU the cache is on by default (64 entries), and cached
    prefixes give their blocks back, oldest first, when a request needs the
    room. The CPU default stays off, since there each entry is a host copy.

- **Concurrent requests share one batched pass and one paged KV pool.**
  - **Batched decode:** a scheduler tick advances every in-flight request
    through `Generator::step_sessions`. On the resident GPU kernel that is one
    fused block call per layer for the whole batch (in chunks of 16).
  - **Batched decode speedup, GTX 1650, prompt 128:** Qwen2.5-0.5B int4 at
    batch 16 went from 62.9 to 79.8 tok/s total. It is bounded because this GPU
    reads each weight row once per sequence either way.
  - **Paged device KV:** each layer's K/V is one pool of 16-token blocks shared
    by every session. A session takes blocks as it grows, reaches them through a
    block table, and returns them when it ends. It used to allocate a
    full-context buffer per layer on its first token.
  - **Pool size:** resident `serve` gives the pool whatever VRAM fits after the
    weights, up to `--max-batch` full contexts. It refuses to start only if one
    full-context request cannot fit; before, it refused whenever `--max-batch`
    full contexts did not.
  - **Admission control:** a request is admitted only when the pools can hold its
    prompt plus `max_tokens` next to what running requests have reserved. It
    waits in the queue otherwise, so the pool never runs dry mid-decode.
- **`--kv-quant f16`: the GPU keeps the KV cache in fp16.**
  - **Before:** device KV was always f32, and `--kv-quant` had no effect on it.
  - **Now:** `f16`, `int8` and `int4` all store device KV as fp16, halving KV
    VRAM. The fit checks reserve accordingly.
  - **Measured:** Qwen2.5-0.5B, 4 sequences at an 8k context: 2.35 GiB -> 1.98
    GiB VRAM in use, with decode speed unchanged.
  - **Output:** greedy output matched f32 KV for 96 of 96 tokens on
    Qwen2.5-0.5B and on Gemma 3 1B.
  - **Default:** `f16`. Pass `--kv-quant f32` for the exact cache (`none` still
    works as an alias). MLA stays f32, and the CPU kernels keep their
    f32/int8/int4 stores.
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
