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
//! so config + tokenizer alone catch the whole class for ~85 MB and no secrets
//! (every source here is ungated, so this runs on fork PRs too).
//!
//! **One fixture per row of the README's support table.** Five rows had none —
//! Llama 3, Mixtral, Qwen2-MoE, Gemma v1 and dense Qwen3 — so those claims rested
//! on nothing. The `llama-2` fixture in particular does not cover Llama 3: they
//! share a table row and almost nothing else, since Llama 3 dropped SentencePiece
//! for a 128k byte-level vocabulary and added `llama3` RoPE scaling.
//!
//! Populate with `.github/fetch-family-fixtures.sh`; each family skips when its
//! directory is absent, so a fresh clone still runs the rest of the suite.

use dlm::forward::cpu::RopeScaling;
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

/// Llama 3: the `llama3` RoPE scaling the README advertises, which nothing
/// asserted until now — the `llama-2` fixture is SentencePiece and covers a
/// different tokenizer shape entirely, so "Llama 2 / 3 / 3.1 / 3.2" was one
/// table row backed by half a test.
///
/// This is the highest-consequence gap of the five: a `rope_scaling` block that
/// parses wrong does not error, it produces fluent nonsense, because the model
/// was *trained* with that correction and running without it silently changes
/// every position. dlm refuses scaling types it does not implement precisely
/// because of that — so the refusal path needs a real config proving the type it
/// *does* implement still parses.
#[test]
fn llama3_family_fixture_parses_rope_scaling() {
    let Some(dir) = fixture("llama-3") else {
        eprintln!("skipping llama-3: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("llama-3 config.json");

    match cfg.rope_scaling {
        Some(RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        }) => {
            assert_eq!(factor, 32.0);
            assert_eq!(low_freq_factor, 1.0);
            assert_eq!(high_freq_factor, 4.0);
            assert_eq!(original_max_position, 8192.0);
        }
        other => panic!("llama-3 must parse as RopeScaling::Llama3, got {other:?}"),
    }

    // Llama 3 dropped SentencePiece for a 128k byte-level vocabulary — the same
    // family row as llama-2, a completely different tokenizer.
    assert_eq!(cfg.vocab_size, 128256);

    let tok = BpeTokenizer::from_dir(&dir).expect("llama-3 tokenizer");
    assert_pieces(
        &tok,
        &["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"],
        "llama-3",
    );
    assert_round_trip(&tok, "llama-3");
}

/// Mixtral layout: `block_sparse_moe.experts.*` with no shared expert — the
/// naming branch that had no real config behind it. DeepSeek covers the
/// `n_routed_experts` spelling and Qwen covers the gated shared expert; this is
/// the third, and the one the README names first.
#[test]
fn mixtral_family_fixture() {
    let Some(dir) = fixture("mixtral") else {
        eprintln!("skipping mixtral: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("mixtral config.json");
    let moe = cfg.moe.expect("num_local_experts must register as MoE");
    assert_eq!(moe.naming, MoeNaming::Mixtral);
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.experts_per_tok, 2);
    assert_eq!(
        moe.shared_intermediate_size, None,
        "Mixtral has no shared expert; inventing one would add weights the \
         checkpoint does not contain"
    );
    assert_eq!(moe.first_k_dense, 0, "every layer is routed");
}

/// Qwen2-MoE: routed experts *plus* a sigmoid-gated shared expert, which is the
/// half of the MoE surface neither Mixtral (no shared expert) nor DeepSeek
/// (ungated shared expert) reaches.
#[test]
fn qwen2_moe_family_fixture() {
    let Some(dir) = fixture("qwen2-moe") else {
        eprintln!("skipping qwen2-moe: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("qwen2-moe config.json");
    let moe = cfg.moe.expect("num_experts must register as MoE");
    assert_eq!(moe.naming, MoeNaming::Qwen);
    assert_eq!(moe.num_experts, 60);
    assert_eq!(moe.experts_per_tok, 4);
    // The routed width is `moe_intermediate_size`, NOT the dense
    // `intermediate_size` (5632) sitting next to it in the same file.
    assert_eq!(moe.moe_intermediate_size, 1408);
    assert_eq!(
        moe.shared_intermediate_size,
        Some(5632),
        "Qwen states the shared expert as a width, unlike DeepSeek's count×width"
    );
    assert!(
        !moe.norm_topk_prob,
        "Qwen1.5-MoE sets norm_topk_prob: false; renormalising anyway would \
         change every routing weight"
    );
}

/// Gemma v1: the same family as gemma-2 and a different block — `(1+w)` RMSNorm,
/// GeGLU, `sqrt(hidden)` embedding scaling, and crucially **no softcapping**.
/// Applying gemma-2's caps here would silently squash every logit.
#[test]
fn gemma1_family_fixture_is_not_gemma2() {
    let Some(dir) = fixture("gemma-1") else {
        eprintln!("skipping gemma-1: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("gemma-1 config.json");
    assert!(cfg.norm_add_one, "Gemma stores RMSNorm weights as (1+w)");
    assert_eq!(
        cfg.attn_logit_softcap, None,
        "softcapping arrived in Gemma2; v1 must not inherit it"
    );
    assert_eq!(cfg.final_logit_softcap, None);
    assert_eq!(
        cfg.sliding_window_pattern, None,
        "v1 has no alternating windows"
    );
    // Embeddings are scaled by sqrt(hidden_size); dropping it changes every logit.
    let scale = cfg
        .embed_scale
        .expect("Gemma scales embeddings by sqrt(hidden)");
    assert!(
        (scale - (cfg.hidden_size as f32).sqrt()).abs() < 1e-3,
        "embed_scale {scale} should be sqrt({})",
        cfg.hidden_size
    );

    let tok = BpeTokenizer::from_dir(&dir).expect("gemma-1 tokenizer");
    assert_round_trip(&tok, "gemma-1");
}

/// Qwen3 dense: `head_dim` is declared explicitly and does **not** equal
/// `hidden_size / num_attention_heads` (128 vs 1024/16 = 64).
///
/// A loader that derives head_dim instead of reading it gets every projection
/// shape wrong here. That is a config-level trap with no tokenizer component,
/// and the reason a dense Qwen3 fixture earns its place next to qwen2.5.
#[test]
fn qwen3_family_fixture_has_explicit_head_dim() {
    let Some(dir) = fixture("qwen3") else {
        eprintln!("skipping qwen3: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("qwen3 config.json");
    assert_eq!(cfg.explicit_head_dim, Some(128));
    assert_ne!(
        cfg.hidden_size / cfg.num_attention_heads,
        128,
        "this fixture is only interesting while derived != declared"
    );
    assert!(cfg.moe.is_none(), "Qwen3-0.6B is dense");

    let tok = BpeTokenizer::from_dir(&dir).expect("qwen3 tokenizer");
    assert_pieces(&tok, &["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"], "qwen3");
    assert_round_trip(&tok, "qwen3");
}

/// Phi-3: a real config for the fused-projection family.
///
/// The weight *mapping* — splitting `qkv_proj` and `gate_up_proj` — is checked in
/// `tests/phi3_fused.rs`, where it can be verified by mutation. This fixture
/// covers the half that lives in `config.json`: that a Phi-3 checkpoint parses at
/// all, and that its declared sliding window survives.
#[test]
fn phi3_family_fixture() {
    let Some(dir) = fixture("phi-3") else {
        eprintln!("skipping phi-3: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("phi-3 config.json");
    // Phi-3-mini-4k declares an odd 2047 rather than a round number; dlm must
    // carry whatever the checkpoint says rather than rounding to a nicer one.
    assert_eq!(cfg.sliding_window, Some(2047));
    assert_eq!(cfg.vocab_size, 32064);
    assert!(cfg.moe.is_none(), "Phi-3-mini is dense");
    // Phi-3-mini is MHA, not GQA — kv heads equal attention heads. The fused QKV
    // split must not assume a narrower K/V.
    assert_eq!(cfg.num_kv_heads, cfg.num_attention_heads);
    assert!(
        cfg.rope_scaling.is_none(),
        "the 4k variant declares no scaling; the 128k one uses longrope, which \
         dlm refuses rather than approximates"
    );

    let tok = BpeTokenizer::from_dir(&dir).expect("phi-3 tokenizer");
    assert_round_trip(&tok, "phi-3");
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
