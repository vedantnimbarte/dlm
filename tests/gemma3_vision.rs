//! Gemma 3 image input against Hugging Face `transformers`, on a tiny random
//! `Gemma3ForConditionalGeneration` (`tests/fixtures/gemma3_vision`, made by
//! `tests/fixtures/make_gemma3_vision.py`).
//!
//! Three stages, each against the reference: the image processor's
//! `pixel_values`, the vision tower and projector's image embeddings, and the
//! log-probabilities of a prompt with the image in it -- which is where the
//! bidirectional attention among an image's tokens shows (a causal mask there
//! moves the image positions' scores and everything after them).

use dlm::generate::ImageEmbeds;
use dlm::model::QuantScheme;
use dlm::vision::{image, VisionTower};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma3_vision")
}

fn reference() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(fixture().join("reference.json")).unwrap()).unwrap()
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// transformers' `pixel_values` for the fixture image, from its bytes.
fn reference_pixels(r: &serde_json::Value) -> Vec<f32> {
    floats(&r["pixels_u8"])
        .into_iter()
        .map(|u| (u - 127.5) / 127.5)
        .collect()
}

fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let (i, d) = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .enumerate()
        .fold((0, 0.0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
    eprintln!("{what}: max abs difference {d:e}");
    assert!(
        d < tol,
        "{what}: off by {d} at {i} (dlm {}, transformers {})",
        got[i],
        want[i]
    );
}

/// Decoding the PNG and resizing it (41x19 to 28x28: shrunk across, stretched
/// down) gives transformers' bytes exactly.
#[test]
fn preprocessing_matches_transformers() {
    let r = reference();
    let bytes = std::fs::read(fixture().join("image.png")).unwrap();
    let got = image::preprocess(&image::decode(&bytes).unwrap(), 28);
    assert_close("pixel_values", &got, &reference_pixels(&r), 1e-6);
}

/// The tower and projector, on the CPU.
#[test]
fn image_embeddings_match_transformers() {
    let r = reference();
    let tower = VisionTower::find(&fixture(), None).unwrap().unwrap();
    let got = tower.encode(&reference_pixels(&r), false).unwrap();
    assert_close("image embeddings", &got, &floats(&r["image_embeds"]), 1e-4);
}

/// The whole prompt on the CPU: every token's log-probability, and the
/// distribution after the last.
#[test]
fn prompt_with_an_image_matches_transformers() {
    let r = reference();
    let ids: Vec<u32> = r["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let tower = VisionTower::find(&fixture(), None).unwrap().unwrap();
    let rows = tower.encode(&reference_pixels(&r), false).unwrap();
    let (config, store) = dlm::loader::open_model(&fixture(), QuantScheme::Fp16).unwrap();
    let generator = dlm::loader::load_generator(&store, &config, ids.len() as u32).unwrap();
    let images = [ImageEmbeds { start: 6, rows }];
    let (got, next) = generator.score_with_images(&ids, &images).unwrap();
    assert_close("token logprobs", &got, &floats(&r["token_logprobs"]), 1e-4);
    assert_close("next logprobs", &next, &floats(&r["next_logprobs"]), 1e-4);
}
