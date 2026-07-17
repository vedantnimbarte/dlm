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

## Known gaps — read before you tag

These are not TODOs; they are the honest limits of what has been verified. A
release is a claim, and these bound it.

- **Only one GPU has ever run this code**: a GTX 1650 (Turing, 4 GB), on Windows.
  Ampere/Ada/Blackwell, and anything with a different warp/SM profile, are
  unexercised.
- **The Linux CUDA path has never been run**, and the release ships a Linux
  `-cuda-static` binary. This is not theoretical: `serve --stream` used to
  over-commit VRAM, which Windows' WDDM driver hides by paging GPU memory to host
  RAM — Linux has no such fallback and would likely have OOM'd outright. That bug
  is fixed, but the fact it survived is evidence the Linux GPU path was never
  exercised. Run at least check 2 above on a Linux box with an NVIDIA GPU.
  (WSL2 does **not** count: its GPU is paravirtualized through the same Windows
  WDDM driver and inherits the same paging behaviour.)
- **Multi-GPU and distributed serving** are untested on real hardware.
- **Models above 3B** are untested; so is any context near the 8192 default.

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
