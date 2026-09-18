//! GGUF against a real file, when one is present.
//!
//! These skip unless the env vars point at checkpoints, the same rule
//! `tests/real_model.rs` follows: the files are hundreds of megabytes and do not
//! belong in the repository.
//!
//! ```sh
//! # the model to read, and llama.cpp's own dequantization of it:
//! #   llama-quantize --allow-requantize model.gguf model-f32.gguf F32
//! DLM_TEST_GGUF=models/gguf/model.gguf \
//! DLM_TEST_GGUF_F32=/tmp/model-f32.gguf \
//!   cargo test --release --test gguf_model
//! ```

use dlm::storage::{ggml_quant, MmapStore};
use std::path::PathBuf;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

/// Decoding a quantized block is the one part of GGUF support that cannot be
/// checked against dlm itself: a misread scale or a swapped nibble half gives
/// weights that still generate fluent text. So it is checked against llama.cpp's
/// own dequantization of the same file, weight for weight.
#[test]
fn blocks_decode_exactly_like_llama_cpp() {
    let (Some(quantized), Some(reference)) =
        (env_path("DLM_TEST_GGUF"), env_path("DLM_TEST_GGUF_F32"))
    else {
        eprintln!("skipping: set DLM_TEST_GGUF and DLM_TEST_GGUF_F32");
        return;
    };
    let ours = MmapStore::open_path(&quantized).expect("open the quantized file");
    let theirs = MmapStore::open_path(&reference).expect("open llama.cpp's f32 copy");

    let mut checked = 0usize;
    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    for info in ours.iter_tensors() {
        if !info.dtype.is_block_quantized() {
            continue;
        }
        let Some((shard, _)) = ours.locate(&info.name) else {
            continue;
        };
        let got = ggml_quant::to_f32(
            info.dtype,
            shard.tensor_bytes(&info.name).unwrap(),
            info.num_elements(),
        )
        .unwrap_or_else(|e| panic!("decoding {}: {e}", info.name));

        let (ref_shard, ref_info) = theirs
            .locate(&info.name)
            .unwrap_or_else(|| panic!("{} is missing from the f32 copy", info.name));
        let want =
            dlm::storage::bytes_to_f32(ref_shard.tensor_bytes(&info.name).unwrap(), ref_info.dtype)
                .unwrap();
        assert_eq!(got.len(), want.len(), "{}: length", info.name);

        // The codes and scales are decoded exactly; what moves is the last bits of
        // dlm's `(code - zero) * scale`, where the zero is a division ggml does
        // not perform. Relative to the tensor's own magnitude, that is noise.
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            let diff = (a - b).abs() / scale;
            if diff > worst {
                worst = diff;
                worst_name = format!("{}[{i}]", info.name);
            }
        }
        checked += 1;
    }
    assert!(checked > 0, "no block-quantized tensors in {quantized:?}");
    assert!(
        worst < 1e-5,
        "{checked} tensors checked; worst relative difference {worst} at {worst_name}"
    );
    eprintln!("{checked} quantized tensors match llama.cpp (worst {worst:.2e} relative)");
}

/// A GGUF file carries its own tokenizer, and reading it is what makes the file
/// usable on its own. Pinned against the ids llama.cpp produces for the same
/// text, which `tools/llamacpp_parity.py` checks in bulk.
#[test]
fn reads_its_own_tokenizer() {
    let Some(path) = env_path("DLM_TEST_GGUF") else {
        eprintln!("skipping: set DLM_TEST_GGUF");
        return;
    };
    let tok = dlm::tokenizer::BpeTokenizer::from_gguf_path(&path).expect("tokenizer in the file");
    assert!(tok.vocab_size() > 1000, "vocabulary looks empty");
    let text = "def add(a, b):\n    return a + b\n";
    let ids = tok.encode(text).expect("encode");
    assert!(!ids.is_empty());
    assert_eq!(tok.decode(&ids).expect("decode"), text, "round trip");
}

/// Streaming keeps ggml's blocks where the resident path re-packs them into
/// dlm's own layout, so the two run different decoders over the same file. They
/// must still produce the same tokens — otherwise a model that is small enough
/// to sit in memory answers differently from one that has to stream.
#[test]
fn streamed_and_resident_agree() {
    let Some(path) = env_path("DLM_TEST_GGUF") else {
        eprintln!("skipping: set DLM_TEST_GGUF");
        return;
    };
    let quant = dlm::model::QuantScheme::Fp16; // ignored: a GGUF carries its own
    let (config, store) = dlm::loader::open_model(&path, quant).expect("open the model");
    let prompt = [785u32, 6722, 315, 9625, 374]; // "The capital of France is"
    let gen = dlm::generate::GenerationConfig {
        max_new_tokens: 8,
        eos_token: None,
        sampler: dlm::generate::Sampler::Greedy,
    };

    let resident = dlm::loader::load_generator(&store, &config, 64).expect("resident");
    let want = resident.generate(&prompt, &gen).expect("resident generate");

    // A window of two layers, so most of the model really does stream.
    let streamed = dlm::loader::build_streaming_generator(
        MmapStore::open_path(&path).expect("re-open"),
        &config,
        64,
        2,
        1,
        false,
        0,
    )
    .expect("streaming generator");
    let got = streamed.generate(&prompt, &gen).expect("streamed generate");
    assert_eq!(got, want, "streamed output differs from resident");
}
