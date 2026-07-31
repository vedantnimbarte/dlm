<!--
Keep this short. The checklist below is not ceremony — each line corresponds to
a way this project has actually shipped a bug.
-->

## What this changes, and why

<!-- The "why" matters more than the "what"; the diff already shows the what. -->

## How it was verified

<!--
Name the command and its result, not "tested locally". A green suite that never
checked the claim is the failure mode this repo has hit most often.

    cargo test
    cargo clippy --all-targets -- -D warnings
    cargo fmt --all -- --check
-->

## Checklist

- [ ] `cargo test`, `cargo clippy --all-targets -- -D warnings`, and
      `cargo fmt --all -- --check` all pass
- [ ] New behaviour has a test that **fails without the change**. If the test
      cannot fail, it proves nothing — mutate the fix and confirm it goes red.
- [ ] If this touches a checkpoint's `config.json` or `tokenizer.json` handling,
      a real fixture for that family is added to
      `.github/fetch-family-fixtures.sh` and `tests/family_fixtures.rs`. Every
      defect found before 0.3.0 lived in those two files, and in-code fixtures
      never touch either.
- [ ] If this changes what dlm claims to support, `README.md` and
      `RELEASING.md`'s ledger are updated to match. A claim nobody has run is a
      hedge, not a feature.
- [ ] If this touches `unsafe`, each block says what the invariant is and **who
      upholds it** — the caller, the driver, or an earlier check.
- [ ] `CHANGELOG.md` updated under `[Unreleased]` for anything user-visible.

<!--
GPU paths: CI has no GPU and never executes a device kernel. If this touches
src/gpu/ or src/forward/*gpu*, say which card you ran
`cargo test --release --features cuda-kernels` (or `rocm-kernels`) on — or say
that you could not, so a reviewer knows it is unverified.
-->
