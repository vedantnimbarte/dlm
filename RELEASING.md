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
#    The same on a real GGUF file: streamed through VRAM against resident, and
#    (with llama.cpp's f32 copy of it) every block against llama.cpp's decode.
#    Run it on a 2-3-bit file (Q2_K, IQ3_M) as well as a Q4_K_M.
DLM_TEST_GGUF=models/gguf/<model>.gguf DLM_TEST_GGUF_F32=<its-f32-copy>.gguf   cargo test --release --features cuda-kernels --test gguf_model

#    And a soak through the HTTP server: many clients, every request kind,
#    abandoned streams; nothing may leak or cross between batched requests.
DLM_SOAK_ROUNDS=300 cargo test --release --features cuda-kernels --test server_soak

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
# And one whose weights do not fit at all, which streams without --stream:
cargo run --release --features cuda-kernels --   serve --model-path models/gguf/<a-7B-Q3_K_M>.gguf --device gpu --port 8000
# expect: a `placement` line saying it streams, then coherent output
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
- **KV cache in host RAM** (a context the card cannot hold) has been run on
  one model, Qwen2.5-1.5B `Q8_0` at a 4k-token history, on the resident GPU
  kernel only. Its speed falls with the history it reads over PCIe each step;
  nothing past 4k has been measured.
- **The newer API features are checked on few models.** Tool calling has a
  prompt format only for ChatML and was run on Qwen2.5-1.5B and Qwen3-0.6B
  (Qwen2.5-0.5B answers instead of calling). `--lora` was checked with one
  adapter, rank 16 on Qwen2.5-0.5B-Instruct, against a torch merge.
  `/v1/embeddings` pools the last token, which suits embedding models; on a
  chat model it is only a usable summary.
- **Llama 2 is untested.** It shares Mistral's SentencePiece tokenizer shape,
  which is verified, so it is expected to work — expected, not demonstrated.

## Tag

```sh
# 1. Bump `version` in Cargo.toml, refresh Cargo.lock (`cargo build`), commit.
# 2. Run the manual GPU checks above.
# 3. Check the release runners (below) are still the ones release.yml names.
# 4. Then:
git tag vX.Y.Z && git push origin vX.Y.Z
```

Watch the release run to the end. Every job must be green and every asset
attached. A failed leg doesn't stop the others: the release still publishes,
missing that asset, and is marked Latest. `install.sh` then 404s on a missing
Apple Silicon build, and on Linux it falls back from a missing CUDA build to
the CPU one.

## Release runners: CI cannot see them

**The release builds run on runner images that nothing else exercises.** CI
never runs `release.yml`, so a green `main` says nothing about them, and
GitHub changes them on its own schedule. v0.4.0 is the example. Not a line of
code had changed since CI passed, and the release still went out without two
of its six builds:

- `macos-14` had been retired. The Apple Silicon job was cancelled before it
  ran a step ("The configuration … is not supported").
- `ubuntu-latest` had moved to Ubuntu 26.04, and NVIDIA's 26.04 repo has no
  `cuda-13-2`. The Linux CUDA job failed installing the toolkit.

So `release.yml` now pins the runner each job depends on, in the way that
fits that job:

| Job | Runner | Why |
|---|---|---|
| Linux CUDA static | `ubuntu-24.04` | The toolkit comes from NVIDIA's per-Ubuntu repo, so the Ubuntu version matters. Move it on once that repo has the pinned CUDA. |
| Apple Silicon | `macos-latest` | Rust's deployment target (11.0) decides which Macs the binary runs on, not the runner's macOS, so following the image costs nothing and survives retirements. |
| Linux CPU | `ubuntu-latest` | Safe to float. The binary's glibc floor comes from the symbols it uses: 2.39 on x86_64 and 2.18 on aarch64 at 0.4.1. A newer image could raise it, so check if Linux users report load errors. |

**Before tagging**, check
[actions/runner-images](https://github.com/actions/runner-images) for images
being retired, and for `-latest` labels that have moved since the last release.
A label moving is as much a change to the release as a commit is.

**If a release run fails anyway,** don't re-run the failed jobs. A re-run uses
`release.yml` as it was at the tag, so it hits the same broken runner. Fix the
workflow, ship a patch release with the same code (0.4.0 → 0.4.1), and mark
the incomplete release as a pre-release with a note pointing to its
replacement. Don't move the tag: by then people may have downloaded its assets.
