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

[Unreleased]: https://github.com/vedantnimbarte/dlm/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/vedantnimbarte/dlm/releases/tag/v0.3.0
