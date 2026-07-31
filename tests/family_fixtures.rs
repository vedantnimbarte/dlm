//! Real `config.json` + `tokenizer.json`, one per supported model family.
//!
//! Every blocker found before 0.3.0 lived in one of those two files:
//!
//! * Gemma2 shipped `hidden_act` *and* `hidden_activation`, and serde rejected
//!   the doubly-matched field — no Gemma2 checkpoint could be loaded at all.
//! * Gemma, Mistral and Llama-2 ship SentencePiece vocabularies under
//!   `"type": "BPE"`, and dlm encoded them with GPT-2 byte-level rules. Gemma and
//!   Llama-2 have a `Ġ` in vocabulary, so they corrupted *silently*; Mistral does
//!   not, so it failed outright.
//! * DeepSeek spells its expert count `n_routed_experts`, so MoE went undetected
//!   and every layer loaded as dense.
//!
//! None of it was visible to the suite, because the in-code fixtures construct
//! `BlockConfig` directly and never parse either file, and the only real
//! checkpoint CI had was Qwen — dense, byte-level BPE, no BOS. That is the one
//! shape which exercises none of these paths.
//!
//! **No weights are downloaded.** These defects were in parsing, and a tokenizer
//! that picks the wrong pieces produces the wrong ids whatever the weights are,
//! so config + tokenizer alone catch the whole class for ~35 MB and no secrets
//! (every source here is ungated, so this runs on fork PRs too).
//!
//! Populate with `.github/fetch-family-fixtures.sh`; each family skips when its
//! directory is absent, so a fresh clone still runs the rest of the suite.

use dlm::model::{ModelConfig, MoeNaming, QuantScheme};
use dlm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

/// The prompt every family tokenizes. Ordinary words with leading spaces — the
/// exact thing SentencePiece and byte-level BPE disagree about.
const PROMPT: &str = "The capital of France is";

fn fixture(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(
        std::env::var("DLM_FAMILY_FIXTURES").unwrap_or_else(|_| "models/fixtures".to_string()),
    )
    .join(name);
    (dir.join("config.json").exists() && dir.join("tokenizer.json").exists()).then_some(dir)
}

/// Assert the tokenizer segments `PROMPT` into exactly `pieces`.
///
/// The expectation is written as piece *strings* and resolved through the
/// checkpoint's own vocabulary, so this cannot pass by the encoder merely
/// agreeing with itself — the ids come from the file, not from dlm.
fn assert_pieces(tok: &BpeTokenizer, pieces: &[&str], family: &str) {
    let mut want: Vec<u32> = Vec::new();
    if let Some(bos) = tok.bos_id() {
        want.push(bos);
    }
    for p in pieces {
        let id = tok
            .id_of(p)
            .unwrap_or_else(|| panic!("{family}: vocabulary has no piece {p:?}"));
        want.push(id);
    }
    let got = tok
        .encode(PROMPT)
        .unwrap_or_else(|e| panic!("{family}: encode failed: {e}"));
    assert_eq!(got, want, "{family}: wrong segmentation of {PROMPT:?}");
}

/// Text must survive a round trip. Catches a decoder that forgets to turn ▁ back
/// into spaces, or strips a prefix it never added.
fn assert_round_trip(tok: &BpeTokenizer, family: &str) {
    let ids = tok.encode(PROMPT).unwrap();
    let back = tok.decode(&ids).unwrap();
    let trimmed = back.trim_start_matches(|c: char| c == '<' || c.is_alphanumeric() || c == '|');
    assert!(
        back.ends_with(PROMPT) || trimmed.contains(PROMPT),
        "{family}: round trip lost the text: {back:?}"
    );
}

// ── SentencePiece families ───────────────────────────────────────────────────
//
// All three signal the convention differently, and each one alone would leave a
// branch of the detector untested.

/// Gemma2: `Replace(" " -> ▁)` normalizer, null pre-tokenizer, and no prepended
/// mark — so `The` stays bare while later words take the ▁.
/// Also the config that could not be parsed at all.
#[test]
fn gemma2_family_fixture() {
    let Some(dir) = fixture("gemma-2") else {
        eprintln!("skipping gemma-2: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("gemma-2 config.json");
    assert_eq!(cfg.attn_logit_softcap, Some(50.0));
    assert_eq!(cfg.final_logit_softcap, Some(30.0));
    assert_eq!(cfg.sliding_window, Some(4096));
    assert_eq!(
        cfg.sliding_window_pattern,
        Some(2),
        "alternating local/global"
    );

    let tok = BpeTokenizer::from_dir(&dir).expect("gemma-2 tokenizer");
    assert_pieces(
        &tok,
        &["The", "▁capital", "▁of", "▁France", "▁is"],
        "gemma-2",
    );
    assert_round_trip(&tok, "gemma-2");
    assert!(
        tok.bos_id().is_some(),
        "gemma-2 is trained with <bos> always present"
    );
}

/// Llama 2: `Sequence[Prepend(▁), Replace(" " -> ▁)]` with a null pre-tokenizer.
/// Its vocabulary *does* contain `Ġ`, so byte-level encoding produced plausible
/// nonsense rather than an error — the failure mode with no symptom.
#[test]
fn llama2_family_fixture() {
    let Some(dir) = fixture("llama-2") else {
        eprintln!("skipping llama-2: fixture absent");
        return;
    };
    let tok = BpeTokenizer::from_dir(&dir).expect("llama-2 tokenizer");
    assert_pieces(
        &tok,
        &["▁The", "▁capital", "▁of", "▁France", "▁is"],
        "llama-2",
    );
    assert_round_trip(&tok, "llama-2");
}

/// Mistral: `Metaspace` pre-tokenizer with a **null normalizer** — the mirror
/// image of Gemma, which is why detection has to read both graphs. Its
/// vocabulary has no `Ġ`, so this one failed loudly.
#[test]
fn mistral_family_fixture() {
    let Some(dir) = fixture("mistral") else {
        eprintln!("skipping mistral: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("mistral config.json");
    // v0.2 dropped the sliding window that v0.1 declared, and the ungated mirror
    // is v0.2 — so the checkpoint says `null` and dlm must not invent one. (The
    // windowed path is covered by the gemma-2 fixture, which declares 4096 with
    // an alternating pattern.)
    assert_eq!(
        cfg.sliding_window, None,
        "v0.2 declares no window; dlm must not add one"
    );

    let tok = BpeTokenizer::from_dir(&dir).expect("mistral tokenizer");
    assert_pieces(
        &tok,
        &["▁The", "▁capital", "▁of", "▁France", "▁is"],
        "mistral",
    );
    assert_round_trip(&tok, "mistral");
}

// ── Non-SentencePiece families ───────────────────────────────────────────────

/// DeepSeek-V2: the MoE config that went undetected. Its tokenizer is
/// byte-level, so this fixture guards the *config* half of the class.
#[test]
fn deepseek_family_fixture() {
    let Some(dir) = fixture("deepseek-v2") else {
        eprintln!("skipping deepseek-v2: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("deepseek config.json");
    let moe = cfg.moe.expect("n_routed_experts must register as MoE");
    assert_eq!(moe.naming, MoeNaming::DeepSeek);
    assert_eq!(moe.num_experts, 64);
    assert_eq!(moe.first_k_dense, 1, "layer 0 is dense, layers 1.. routed");
    // A shared-expert *count* (2) times the routed width (1408), not a width.
    assert_eq!(moe.shared_intermediate_size, Some(2816));
    assert!(
        cfg.mla.is_some(),
        "DeepSeek-V2 uses Multi-head Latent Attention"
    );

    let tok = BpeTokenizer::from_dir(&dir).expect("deepseek tokenizer");
    assert_round_trip(&tok, "deepseek-v2");
}

/// Qwen is the control: genuinely byte-level BPE with `add_bos_token: false`.
/// If SentencePiece detection ever over-fires, this is what catches it — the
/// regression that would silently break every model that was working.
#[test]
fn qwen_family_fixture_stays_byte_level() {
    let Some(dir) = fixture("qwen2.5") else {
        eprintln!("skipping qwen2.5: fixture absent");
        return;
    };
    let tok = BpeTokenizer::from_dir(&dir).expect("qwen tokenizer");
    assert_eq!(tok.bos_id(), None, "Qwen sets add_bos_token: false");
    // Byte-level: the space rides along as `Ġ`, and there is no ▁ piece at all.
    assert_pieces(
        &tok,
        &["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"],
        "qwen2.5",
    );
    assert_round_trip(&tok, "qwen2.5");
}
