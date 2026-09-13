//! Gemma 3 on the **real** checkpoint.
//!
//! Gemma 3's differences from Gemma 2 are all invisible to a short prompt: the
//! 5:1 local/global layer pattern and the separate RoPE base of the local layers
//! only change anything once the context outgrows the 512-token window, and a
//! short greedy answer comes out right either way. So besides the answer, this
//! scores a passage longer than the window and bounds its cross-entropy: running
//! every layer as a global one, or the local layers with the global RoPE base,
//! still produces fluent text but a measurably worse fit.
//!
//! Skipped when the checkpoint is absent; fetch it with
//! `dlm pull unsloth/gemma-3-1b-it --local-dir models/gemma-3-1b`, or point
//! `DLM_GEMMA3_MODEL` at one.

use dlm::generate::{GenerationConfig, Sampler};
use dlm::loader::load_model_parts;
use dlm::model::{ModelConfig, QuantScheme};
use dlm::server::engine::{ChatMessage, ChatTemplate};
use dlm::storage::MmapStore;
use dlm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(
        std::env::var("DLM_GEMMA3_MODEL").unwrap_or_else(|_| "models/gemma-3-1b".to_string()),
    );
    dir.join("model.safetensors").exists().then_some(dir)
}

/// A passage of plain technical prose (~560 tokens), a little longer than the
/// local layers' window on its own.
#[cfg(feature = "cuda-kernels")]
const PASSAGE: &str = "A layer-streaming inference engine keeps only a small window of transformer \
blocks in GPU memory and moves the rest across the bus as the computation reaches them. The idea is \
simple, but the details decide whether it is useful. The first detail is the order of work. A prompt \
can be processed one token at a time, sweeping every layer for each token, or one layer at a time, \
sweeping every token through each layer. Both orders compute exactly the same numbers, because a \
token at a given layer only reads the keys and values that earlier tokens wrote at that same layer. \
The second order, however, loads each layer once for the whole prompt instead of once per token, and \
when loading a layer means copying hundreds of megabytes across a bus, that difference dominates \
everything else. The second detail is precision. Weights stored as sixteen-bit floats can be \
quantized to eight or four bits when they are loaded, which shrinks each layer and often lets the \
whole model fit, so that nothing needs to stream at all. Quantization is lossy, and the loss is not \
uniform: the matrices that every token passes through, such as the output head, are more sensitive \
than the rest, and it is common to keep them at a higher precision than the blocks. The third detail \
is attention itself. Every generated token attends over the keys and values of all the tokens before \
it, so the cost of attention grows with the length of the context. Some model families limit this by \
letting most layers attend only to a recent window of tokens while a few layers attend to everything. \
Those global layers carry information across long distances, and the windowed layers handle local \
structure cheaply. Getting the assignment wrong does not produce an error. The model still writes \
grammatical text, but it forgets things it should remember, or it confuses the order of words, and \
the mistake only becomes visible on inputs longer than the window. The fourth detail is position. \
Rotary position embeddings rotate the query and key vectors by angles that depend on each token's \
position, with a base frequency that controls how quickly the angles change across dimensions. A \
model trained with one base for its windowed layers and another for its global layers must be run \
with both, layer by layer, because the attention patterns it learned depend on those exact angles. \
The last detail is verification. It is not enough for an implementation to agree with itself, for \
example by checking that a streamed run matches a resident one, because both can share the same \
mistake. It has to be checked against something independent: a reference implementation, a \
measurement of how well the model predicts ordinary text, or a question whose answer requires \
remembering something from far back in the context. Each check catches a different class of error, \
and a careful engine uses all of them before it claims that a model family is supported.";

#[test]
fn gemma3_answers_a_factual_question() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no Gemma 3 checkpoint");
        return;
    };
    let config = ModelConfig::from_path(&dir, QuantScheme::Fp16).unwrap();
    assert_eq!(config.sliding_window_pattern, Some(6));
    assert_eq!(config.rope_local_theta, Some(10_000.0));
    let tokenizer = BpeTokenizer::from_dir(&dir).unwrap();
    let template = ChatTemplate::read_jinja(&dir)
        .and_then(|j| ChatTemplate::detect(&j))
        .expect("Gemma 3's chat template should be recognized");
    let prompt = template.apply(&[ChatMessage {
        role: "user".into(),
        content: "What is the capital of France? Answer in one word.".into(),
    }]);
    let ids = tokenizer.encode(&prompt).unwrap();
    let store = MmapStore::open_dir(&dir).unwrap();
    let g = load_model_parts(&store, &config, 256)
        .unwrap()
        .into_cpu_generator()
        .unwrap();
    let out = g
        .generate(
            &ids,
            &GenerationConfig {
                max_new_tokens: 4,
                eos_token: tokenizer.id_of("<end_of_turn>"),
                sampler: Sampler::Greedy,
            },
        )
        .unwrap();
    let text = tokenizer.decode(&out).unwrap();
    assert!(text.contains("Paris"), "expected Paris, got {text:?}");
}

/// Cross-entropy of [`PASSAGE`], scored on the GPU (the CPU takes minutes).
#[cfg(feature = "cuda-kernels")]
#[test]
fn gemma3_long_context_fit() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no Gemma 3 checkpoint");
        return;
    };
    let config = ModelConfig::from_path(&dir, QuantScheme::Fp16).unwrap();
    let tokenizer = BpeTokenizer::from_dir(&dir).unwrap();
    // Twice over: the second copy is only predictable by recalling the first
    // from beyond the window, which only the global layers can do.
    let ids = tokenizer
        .encode(&format!("{PASSAGE}\n\n{PASSAGE}"))
        .unwrap();
    assert!(ids.len() > 1024, "passage is {} tokens", ids.len());
    let store = MmapStore::open_dir(&dir).unwrap();
    let g = load_model_parts(&store, &config, 1536)
        .unwrap()
        .into_gpu_generator()
        .unwrap();
    let lp = g.score(&ids).unwrap();
    let nll = |s: &[f32]| -s.iter().map(|&x| x as f64).sum::<f64>() / s.len() as f64;
    let (all, late) = (nll(&lp), nll(&lp[512..]));
    eprintln!(
        "gemma3 NLL: all {all:.4}, past the window {late:.4} ({} tokens)",
        ids.len()
    );
    // Measured on the GTX 1650: 1.80 overall and 0.33 past the window. The local
    // layers rotated with the global base score 3.18 / 2.05, and the Gemma 2
    // layer rule (windowing layers 0, 6, 12, ... instead of all but 5, 11, ...)
    // scores 3.35 / 2.51, so these bounds sit well clear of both.
    assert!(all < 2.3, "cross-entropy {all:.3} over the whole passage");
    assert!(
        late < 0.8,
        "cross-entropy {late:.3} past the window: recall is broken"
    );

    // The streamed GPU kernel resolves windows and RoPE bases on its own, so it
    // must score the passage the same, with most of the model off the card.
    drop(g);
    let store = MmapStore::open_dir(&dir).unwrap();
    // A host RAM cache, so a streamed layer is read from the checkpoint once
    // rather than on every decode miss: this checks correctness, not disk speed.
    let streamed =
        dlm::loader::build_streaming_gpu_generator(store, &config, 1536, 8, 3 << 30, 0).unwrap();
    let streamed_all = nll(&streamed.score(&ids).unwrap());
    assert!(
        (streamed_all - all).abs() < 1e-3,
        "streamed GPU scores {streamed_all:.4}, resident {all:.4}"
    );
}

/// The multimodal 4B checkpoint, seen as its text model.
fn model_dir_4b() -> Option<PathBuf> {
    let dir = PathBuf::from(
        std::env::var("DLM_GEMMA3_4B_MODEL").unwrap_or_else(|_| "models/gemma-3-4b".to_string()),
    );
    dir.join("config.json").exists().then_some(dir)
}

/// Gemma 3 4B stores its language model under `language_model.` beside a SigLIP
/// vision tower whose encoder blocks are also named `...layers.N...`. The store
/// must present exactly the text model: 34 transformer layers, the text names,
/// and no vision tensor anywhere -- counted as layers they would corrupt the
/// catalog, and counted as pinned they would be planned into VRAM.
#[test]
fn gemma3_4b_is_read_as_its_text_model() {
    let Some(dir) = model_dir_4b() else {
        eprintln!("skipping: no Gemma 3 4B checkpoint");
        return;
    };
    let config = ModelConfig::from_path(&dir, QuantScheme::Fp16).unwrap();
    assert_eq!((config.num_layers, config.hidden_size), (34, 2560));
    let store = MmapStore::open_dir(&dir).unwrap();
    assert!(store.locate("model.embed_tokens.weight").is_some());
    assert!(store
        .locate("model.layers.33.self_attn.q_norm.weight")
        .is_some());
    assert!(
        store
            .iter_tensors()
            .all(|t| !t.name.contains("vision") && !t.name.contains("multi_modal")),
        "vision tensors leaked into the text model's index"
    );
    let catalog = dlm::storage::LayerCatalog::build(&store);
    assert_eq!(
        catalog.num_layers(),
        34,
        "vision encoder blocks counted as layers"
    );
}

/// Cross-entropy of the doubled [`PASSAGE`] on the 4B, in bf16 (so the number
/// measures the forward pass, not quantization), streamed through the GPU since
/// the weights are twice the card.
#[cfg(feature = "cuda-kernels")]
#[test]
fn gemma3_4b_long_context_fit() {
    let Some(dir) = model_dir_4b() else {
        eprintln!("skipping: no Gemma 3 4B checkpoint");
        return;
    };
    let config = ModelConfig::from_path(&dir, QuantScheme::Fp16).unwrap();
    let tokenizer = BpeTokenizer::from_dir(&dir).unwrap();
    let ids = tokenizer
        .encode(&format!("{PASSAGE}\n\n{PASSAGE}"))
        .unwrap();
    let store = MmapStore::open_dir(&dir).unwrap();
    let g = dlm::loader::build_streaming_gpu_generator(store, &config, 1536, 8, 0, 0).unwrap();
    let t = std::time::Instant::now();
    let lp = g.score(&ids).unwrap();
    let nll = |s: &[f32]| -s.iter().map(|&x| x as f64).sum::<f64>() / s.len() as f64;
    let (all, late) = (nll(&lp), nll(&lp[1024..]));
    eprintln!(
        "gemma3 4B NLL: all {all:.4}, past the window {late:.4} ({} tokens, {:?})",
        ids.len(),
        t.elapsed()
    );
    // Measured on the GTX 1650: 1.53 overall, ~0 past the window (a 4B copies
    // the repeat). Scaling the local layers' RoPE by the global layers' x8 scores
    // 5.01. Dropping the x8 from the global layers moves this passage by under
    // 0.001 -- linear scaling of a 1M base only matters far past 1,126 tokens --
    // so that case is pinned by `gpu_gemma3_layers_match_cpu` instead.
    assert!(all < 1.9, "cross-entropy {all:.3} over the whole passage");
    assert!(
        late < 0.3,
        "cross-entropy {late:.3} past the window: recall is broken"
    );
}
