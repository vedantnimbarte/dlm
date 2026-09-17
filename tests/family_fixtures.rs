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

/// GPT-2: the one family that is not Llama-descended.
///
/// Its config spells every core dimension differently (`n_embd`, `n_head`,
/// `n_layer`) and ships **both** `n_ctx` and `n_positions` -- aliasing both
/// makes serde reject the file as a duplicate field, which is exactly the bug
/// that made every Gemma2 config unparseable. This fixture is what stops that
/// being reintroduced.
#[test]
fn gpt2_family_fixture_parses_its_own_key_names() {
    let Some(dir) = fixture("gpt2") else {
        eprintln!("skipping gpt2: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::F32).expect("gpt2 config.json");
    assert_eq!(cfg.hidden_size, 768, "n_embd");
    assert_eq!(cfg.num_attention_heads, 12, "n_head");
    assert_eq!(cfg.num_layers, 12, "n_layer");
    assert_eq!(cfg.vocab_size, 50257);
    // GPT-2 predates GQA: kv heads equal attention heads.
    assert_eq!(cfg.num_kv_heads, cfg.num_attention_heads);
    // LayerNorm, learned positions, ungated MLP -- the three ways its block
    // differs from every other family dlm supports.
    assert_eq!(cfg.norm_kind, dlm::forward::cpu::NormKind::Layer);
    assert!(cfg.learned_positions, "GPT-2 uses wpe, not RoPE");
    assert_eq!(cfg.ffn_kind, dlm::forward::cpu::FfnKind::Plain);
    assert!(!cfg.parallel_residual, "that is Falcon, not GPT-2");

    let tok = BpeTokenizer::from_dir(&dir).expect("gpt2 tokenizer");
    assert_pieces(&tok, &["The", "Ġcapital", "Ġof", "ĠFrance", "Ġis"], "gpt2");
    assert_round_trip(&tok, "gpt2");
}

/// Falcon: parallel attention/FFN, multi-query, LayerNorm, ungated MLP.
///
/// `multi_query` is a *flag*, not a count -- a loader that reads
/// `num_key_value_heads` and finds nothing would fall back to 71 KV heads
/// instead of 1 and mis-shape every K/V projection.
#[test]
fn falcon_family_fixture() {
    let Some(dir) = fixture("falcon") else {
        eprintln!("skipping falcon: fixture absent");
        return;
    };
    let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("falcon config.json");
    assert_eq!(cfg.num_attention_heads, 71);
    assert_eq!(cfg.num_kv_heads, 1, "multi_query means one shared KV head");
    assert!(cfg.parallel_residual, "falcon-7b sets parallel_attn");
    assert_eq!(cfg.norm_kind, dlm::forward::cpu::NormKind::Layer);
    assert_eq!(cfg.ffn_kind, dlm::forward::cpu::FfnKind::Plain);
    assert!(!cfg.learned_positions, "Falcon uses RoPE, not wpe");

    let tok = BpeTokenizer::from_dir(&dir).expect("falcon tokenizer");
    assert_round_trip(&tok, "falcon");
}

/// The two Falcon variants dlm refuses rather than mis-decodes. Both would
/// otherwise load and emit fluent, wrong text.
#[test]
fn falcon_alibi_and_new_decoder_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let base = r#""model_type":"falcon","hidden_size":64,"num_attention_heads":8,
                  "num_hidden_layers":2,"vocab_size":128,"layer_norm_epsilon":1e-5"#;
    for (extra, want) in [
        (r#""alibi":true"#, "ALiBi"),
        (
            r#""new_decoder_architecture":true"#,
            "new_decoder_architecture",
        ),
    ] {
        std::fs::write(
            dir.path().join("config.json"),
            format!("{{{base},{extra}}}"),
        )
        .unwrap();
        let err = ModelConfig::from_path(dir.path(), QuantScheme::Fp16)
            .expect_err("must be refused, not silently mis-decoded");
        assert!(format!("{err}").contains(want), "{err}");
    }
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

/// `--chat-template auto` against each family's real `tokenizer_config.json`:
/// the fingerprint must pick the format the model was trained on, and that
/// format's end-of-turn marker must be a real token, or generation would run
/// past the turn. Families whose fixture ships no template (base models, and
/// mirrors that predate chat templates) must detect as none, not as a guess.
#[test]
fn chat_template_auto_detection_per_family() {
    use dlm::server::engine::ChatTemplate;
    let cases = [
        ("qwen2.5", Some(ChatTemplate::ChatMl)),
        ("qwen3", Some(ChatTemplate::ChatMl)),
        ("qwen2-moe", Some(ChatTemplate::ChatMl)),
        ("mixtral", Some(ChatTemplate::ChatMl)), // Nous-Hermes fine-tune: ChatML
        ("llama-3", Some(ChatTemplate::Llama3)),
        ("gemma-1", Some(ChatTemplate::Gemma)),
        ("gemma-2", Some(ChatTemplate::Gemma)),
        ("phi-3", Some(ChatTemplate::Phi3)),
        ("deepseek-v2", Some(ChatTemplate::DeepSeek)),
        ("llama-2", None),
        ("mistral", None),
        ("gpt2", None),
        ("falcon", None),
    ];
    for (family, want) in cases {
        let Some(dir) = fixture(family) else {
            eprintln!("skipping {family}: fixture absent");
            continue;
        };
        let got = ChatTemplate::read_jinja(&dir).and_then(|j| ChatTemplate::detect(&j));
        assert_eq!(got, want, "{family}: wrong chat template detected");
        if let Some(eot) = got.and_then(|t| t.end_of_turn()) {
            let tok = BpeTokenizer::from_dir(&dir).expect("tokenizer");
            assert!(
                tok.id_of(eot).is_some(),
                "{family}: end-of-turn {eot:?} is not a token"
            );
        }
    }
}

/// Gemma 3: Gemma 2's norm layout and (1+w) norms, plus a 5:1 local/global layer
/// pattern whose local layers use their own RoPE base. The 1B states the pattern
/// as `sliding_window_pattern: 6`; the 270M as a `layer_types` list; the 4B nests
/// the whole text model in a multimodal config's `text_config`. All must resolve
/// to the same rule, and to layers 5, 11, 17, ... being the global ones.
#[test]
fn gemma3_family_fixtures() {
    use dlm::server::engine::ChatTemplate;
    for (family, window) in [
        ("gemma-3", 512),
        ("gemma-3-270m", 512),
        ("gemma-3-4b", 1024),
    ] {
        let Some(dir) = fixture(family) else {
            eprintln!("skipping {family}: fixture absent");
            return;
        };
        let cfg = ModelConfig::from_path(&dir, QuantScheme::Fp16)
            .unwrap_or_else(|e| panic!("{family} config.json: {e}"));
        assert_eq!(cfg.sliding_window_pattern, Some(6), "{family}");
        assert_eq!(cfg.sliding_window, Some(window), "{family}");
        assert_eq!(cfg.rope_local_theta, Some(10_000.0), "{family}");
        assert_eq!(cfg.rope_theta, 1_000_000.0, "{family}");
        assert!(cfg.gemma2_norms && cfg.norm_add_one, "{family}");
        assert_eq!(
            cfg.attn_logit_softcap, None,
            "{family}: Gemma 3 dropped softcapping"
        );
        assert_eq!(cfg.final_logit_softcap, None, "{family}");
        assert_eq!(cfg.explicit_head_dim, Some(256), "{family}");

        let tok = BpeTokenizer::from_dir(&dir).expect("gemma-3 tokenizer");
        assert!(tok.bos_id().is_some(), "{family}: trained with <bos>");
        assert!(tok.id_of("<end_of_turn>").is_some(), "{family}");
        assert_round_trip(&tok, family);

        let template = ChatTemplate::read_jinja(&dir).and_then(|j| ChatTemplate::detect(&j));
        assert_eq!(template, Some(ChatTemplate::Gemma), "{family}");
    }
}

/// Byte-level tokenizers split text with their model's own regex before merging:
/// runs of spaces become one token, a line break ends its piece, digits split
/// per the pattern. dlm used to split at spaces alone, and on 411 mixed strings
/// matched Hugging Face `tokenizers` for only 58% (Qwen2.5), 54% (Llama 3) and
/// 71% (GPT-2) of them. The expected ids here come from `tokenizers` itself
/// (`encode(text, add_special_tokens=True)`).
#[test]
fn byte_level_tokenizers_split_like_their_regex() {
    const CODE: &str = "def f(x):\n    return x**2  # square\n";
    const PROSE: &str = "I don't think it's 12345 dollars.";
    let cases: [(&str, &[u32], &[u32]); 3] = [
        (
            "qwen2.5",
            &[
                750, 282, 2075, 982, 262, 470, 856, 334, 17, 220, 671, 9334, 198,
            ],
            &[
                40, 1513, 944, 1744, 432, 594, 220, 16, 17, 18, 19, 20, 11192, 13,
            ],
        ),
        (
            "llama-3",
            &[
                128000, 755, 282, 2120, 997, 262, 471, 865, 334, 17, 220, 674, 9518, 198,
            ],
            &[
                128000, 40, 1541, 956, 1781, 433, 596, 220, 4513, 1774, 11441, 13,
            ],
        ),
        (
            "gpt2",
            &[
                4299, 277, 7, 87, 2599, 198, 220, 220, 220, 1441, 2124, 1174, 17, 220, 1303, 6616,
                198,
            ],
            &[40, 836, 470, 892, 340, 338, 17031, 2231, 5054, 13],
        ),
    ];
    for (family, code, prose) in cases {
        let Some(dir) = fixture(family) else {
            eprintln!("skipping {family}: fixture absent");
            continue;
        };
        let tok = BpeTokenizer::from_dir(&dir).unwrap_or_else(|e| panic!("{family}: {e}"));
        assert_eq!(tok.encode(CODE).unwrap(), code, "{family}: code");
        assert_eq!(tok.encode(PROSE).unwrap(), prose, "{family}: prose");
    }
}
