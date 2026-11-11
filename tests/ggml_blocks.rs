//! Every ggml block type dlm decodes, against llama.cpp's own dequantization.
//!
//! `tests/fixtures/ggml_blocks.gguf` holds random blocks of each type next to
//! the f32 values llama.cpp's reference decoder (gguf-py) gives them;
//! `tests/fixtures/make_ggml_blocks.py` wrote it. A misread field in a block
//! layout still gives fluent text, so this is the check that the layouts are
//! right, and it runs without a model.

use dlm::storage::{bytes_to_f32, ggml_quant, Dtype, MmapStore};

#[test]
fn every_block_type_decodes_like_llama_cpp() {
    let store = MmapStore::open_path("tests/fixtures/ggml_blocks.gguf").unwrap();
    let mut seen = Vec::new();
    for info in store.iter_tensors() {
        if !info.dtype.is_block_quantized() {
            continue;
        }
        let bytes = store.tensor_bytes(&info.name).unwrap();
        let got = ggml_quant::to_f32(info.dtype, bytes, info.num_elements()).unwrap();
        let reference = format!("{}.f32", info.name);
        let (_, ref_info) = store.locate(&reference).unwrap();
        let want = bytes_to_f32(store.tensor_bytes(&reference).unwrap(), ref_info.dtype).unwrap();
        assert_eq!(got.len(), want.len(), "{}", info.name);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                "{} weight {i}: dlm {g}, llama.cpp {w}",
                info.name
            );
        }
        seen.push(info.dtype);
    }
    for dtype in [
        Dtype::Q2K,
        Dtype::Q3K,
        Dtype::Q4K,
        Dtype::IQ2XXS,
        Dtype::IQ2XS,
        Dtype::IQ2S,
        Dtype::IQ3XXS,
        Dtype::IQ3S,
        Dtype::IQ4NL,
        Dtype::IQ4XS,
    ] {
        assert!(seen.contains(&dtype), "the fixture has no {dtype:?}");
    }
}
