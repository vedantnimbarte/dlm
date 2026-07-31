# Contributing to dlm

Thanks for your interest in contributing!

## Getting started

1. Fork and clone the repo.
2. Build: `cargo build --release` (add `--features cuda-kernels` if you have an NVIDIA GPU and the CUDA toolkit).
3. Test: `cargo test`. GPU-dependent tests only run with `--features cuda-kernels` on a machine with a CUDA GPU.

## Making changes

- Open an issue first for anything non-trivial so we can discuss the approach.
- Keep PRs focused — one change per PR.
- Run `cargo fmt` and `cargo clippy` before pushing.
- Add or update tests for behavior changes (integration tests live in `tests/`).
- Use clear commit messages (`fix(stream): ...`, `feat(quant): ...`, `docs: ...` — match the existing history).

## Adding or updating a model family

Model support is **continuous, not scheduled**. A new checkpoint family lands on
someone else's release calendar, so waiting for a dlm milestone to react means
being wrong for months. What decides the work is not the model's size but its
shape:

| Tier | What it is | Effort | When |
|---|---|---|---|
| **A** | A family dlm already claims, with no fixture behind the claim | hours | immediately — it is a debt, not a feature |
| **B** | A **new version** of a supported architecture (Gemma 3 after Gemma 2, a new Qwen, a new Llama) | days to weeks | as it ships, outside the release cycle |
| **C** | A **genuinely new block structure** (Falcon's parallel attention, GPT-2's learned positions) | a release | planned into a milestone |

Tier B is the one that bites. A new version usually parses — until it does not,
or worse, until it parses *wrongly*. Every defect found before 0.3.0 lived in a
checkpoint's `config.json` or `tokenizer.json`, not in the engine:

- Gemma2 shipped `hidden_act` **and** `hidden_activation`; serde rejected the
  doubly-matched field and no Gemma2 checkpoint would load at all.
- Gemma, Mistral and Llama-2 ship SentencePiece vocabularies under
  `"type": "BPE"`. dlm encoded them with byte-level rules. Gemma and Llama-2 have
  a `Ġ` in vocabulary, so they corrupted **silently**.
- DeepSeek spells its expert count `n_routed_experts`, so MoE went undetected and
  every layer loaded dense.

None was visible to a green suite. So:

**Every family in the README's support table must have a fixture in
`.github/fetch-family-fixtures.sh` and `tests/family_fixtures.rs`.** A family
without one is a family nobody has actually loaded — the claim is a hedge. If you
add a row to that table, add the fixture in the same PR.

The checklist for a new family:

1. Add a real `config.json` + `tokenizer.json` fixture. **No weights** — these are
   parsing defects, and a tokenizer that picks the wrong pieces yields the wrong
   ids whatever the weights are.
2. Use an **ungated** source, so the job needs no secrets and runs on fork PRs.
   Where the canonical repo is gated, a public mirror of the byte-identical
   tokenizer is fine — several entries already do this.
3. Assert something **specific to that family**, not that it parses. Write
   tokenizer expectations as piece *strings* resolved through the checkpoint's own
   vocabulary, so a pass cannot mean the encoder merely agrees with itself.
4. **Verify by mutation.** Revert the code the fixture guards and confirm the test
   goes red. A test that cannot fail proves nothing.
5. Update the README table and `RELEASING.md`'s ledger together. A claim nobody
   has run belongs in the ledger, not the table.

If dlm refuses a checkpoint, that is working as designed — it fails with a clear
`UnknownTensor` error rather than producing garbage. Open a
[model support issue](https://github.com/vedantnimbarte/dlm/issues/new?template=model_support.yml)
with the `config.json` and we can tell you which tier it is.

## Reporting bugs

Open an issue with your OS, GPU, dlm version, the exact command you ran, and the full output.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 license](LICENSE.md).
