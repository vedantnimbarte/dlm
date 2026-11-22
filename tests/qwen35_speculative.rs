//! Speculative decoding on a hybrid (Qwen3.5/3.6) model, whose linear-attention
//! layers keep a recurrent state that cannot be cut back like K/V: a rejected
//! round restores the state from before it and replays what was accepted.
//! Greedy output must be plain greedy decoding's, token for token.
//!
//! The fixture's MTP head is random, so most of its guesses are rejected: the
//! rollback runs almost every round.

use dlm::loader::{load_generator, load_mtp, open_model};
use dlm::speculative::SpeculativeSession;
use dlm::{GenerationConfig, QuantScheme, Sampler};
use std::path::Path;

const N: usize = 24;

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_moe")
}

fn run(
    mut s: SpeculativeSession<'_, impl dlm::forward::ComputeKernel>,
) -> (Vec<u32>, usize, usize) {
    let mut out = Vec::new();
    while out.len() < N {
        out.extend(s.step(N - out.len()).unwrap());
    }
    out.truncate(N);
    (out, s.proposed(), s.accepted())
}

#[test]
fn speculation_on_a_recurrent_model_matches_plain_greedy() {
    let (config, store) = open_model(&fixture(), QuantScheme::Fp16).unwrap();
    let g = load_generator(&store, &config, 128).unwrap();
    let prompt: Vec<u32> = vec![3, 17, 5, 9, 3, 17, 5, 9, 3, 17];
    let plain = g
        .generate(
            &prompt,
            &GenerationConfig {
                max_new_tokens: N,
                eos_token: None,
                sampler: Sampler::Greedy,
            },
        )
        .unwrap();

    let mtp = load_mtp(&store, &config)
        .unwrap()
        .expect("the fixture ships an MTP head");
    for gamma in [1, 2, 3] {
        let s = SpeculativeSession::mtp(&g, &mtp, gamma, &prompt, Sampler::Greedy).unwrap();
        let (out, proposed, accepted) = run(s);
        assert_eq!(out, plain, "MTP gamma {gamma}");
        assert!(proposed > 0, "MTP gamma {gamma}: nothing proposed");
        assert!(
            accepted < proposed,
            "the random head should miss: rollback untested"
        );
    }

    let s = SpeculativeSession::ngram(&g, 3, &prompt, Sampler::Greedy).unwrap();
    let (out, proposed, _) = run(s);
    assert_eq!(out, plain, "n-gram");
    assert!(
        proposed > 0,
        "n-gram: the prompt repeats, so lookup proposes"
    );
}

/// The same on the GPU, whose linear-attention state lives in device memory:
/// a checkpoint downloads it, a rollback uploads it again.
#[cfg(feature = "cuda-kernels")]
fn gpu_case(
    g: &dlm::Generator<impl dlm::forward::ComputeKernel>,
    store: &dlm::storage::MmapStore,
    config: &dlm::ModelConfig,
    kind: &str,
) {
    let prompt: Vec<u32> = vec![3, 17, 5, 9, 3, 17, 5, 9, 3, 17];
    let plain = g
        .generate(
            &prompt,
            &GenerationConfig {
                max_new_tokens: N,
                eos_token: None,
                sampler: Sampler::Greedy,
            },
        )
        .unwrap();
    let mtp = load_mtp(store, config)
        .unwrap()
        .expect("the fixture ships an MTP head");
    for gamma in [1, 3] {
        let s = SpeculativeSession::mtp(g, &mtp, gamma, &prompt, Sampler::Greedy).unwrap();
        let (out, proposed, accepted) = run(s);
        assert_eq!(out, plain, "{kind}: MTP gamma {gamma}");
        assert!(
            proposed > 0 && accepted < proposed,
            "{kind}: gamma {gamma}: rollback untested"
        );
    }
    let s = SpeculativeSession::ngram(g, 3, &prompt, Sampler::Greedy).unwrap();
    assert_eq!(run(s).0, plain, "{kind}: n-gram");
}

/// The same on the GPU, resident and streamed (a 2-layer window, a cache of a
/// few experts), whose linear-attention state lives in device memory: a
/// checkpoint downloads it, a rollback uploads it again.
#[cfg(feature = "cuda-kernels")]
#[test]
fn gpu_speculation_on_a_recurrent_model_matches_plain_greedy() {
    let (config, store) = open_model(&fixture(), QuantScheme::Fp16).unwrap();
    let g = dlm::loader::load_model_parts(&store, &config, 128)
        .unwrap()
        .into_gpu_generator()
        .unwrap();
    gpu_case(&g, &store, &config, "resident");
    let (config, store2) = open_model(&fixture(), QuantScheme::Fp16).unwrap();
    let streamed =
        dlm::loader::build_streaming_gpu_generator(store2, &config, 128, 2, 0, 3 * 4096, false)
            .unwrap();
    gpu_case(&streamed, &store, &config, "streamed");
}
