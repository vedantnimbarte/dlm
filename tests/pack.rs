//! `dlm pack` round trip: a pack scores exactly what its source does under the
//! same `--quant`, since loading it hands back the bytes the loader produced.

use dlm::model::QuantScheme;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn score(dir: &Path, quant: QuantScheme, ids: &[u32]) -> Vec<f32> {
    let (config, store) = dlm::loader::open_model(dir, quant).unwrap();
    dlm::loader::load_generator(&store, &config, ids.len() as u32)
        .unwrap()
        .score(ids)
        .unwrap()
}

fn round_trip(name: &str, quant: QuantScheme) {
    let src = fixture(name);
    let out =
        std::env::temp_dir().join(format!("dlm-pack-{name}-{quant:?}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    dlm::pack::write_pack(&src, &out, quant, |_, _| {}).unwrap();

    let store = dlm::storage::MmapStore::open_dir(&out).unwrap();
    assert_eq!(dlm::pack::pack_scheme(&store), Some(quant), "{name}");
    assert!(
        !dlm::pack::packed_names(&store).is_empty(),
        "{name}: nothing was packed"
    );
    assert_eq!(
        dlm::loader::checkpoint_scheme(&store).unwrap(),
        quant,
        "{name}: a pack loads at its own precision by default"
    );

    let ids: Vec<u32> = (0..12).map(|i| (i * 7 + 3) % 60).collect();
    assert_eq!(
        score(&src, quant, &ids),
        score(&out, quant, &ids),
        "{name}: the pack scores differently from its source"
    );
    std::fs::remove_dir_all(&out).unwrap();
}

#[test]
fn a_pack_scores_like_its_source() {
    for name in ["deepseek_v3", "qwen35_moe"] {
        round_trip(name, QuantScheme::Int4);
        round_trip(name, QuantScheme::Int8);
    }
}

#[test]
fn a_pack_refuses_another_precision_and_gguf() {
    let out = std::env::temp_dir().join(format!("dlm-pack-refuse-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    dlm::pack::write_pack(&fixture("deepseek_v3"), &out, QuantScheme::Int4, |_, _| {}).unwrap();
    let (config, store) = dlm::loader::open_model(&out, QuantScheme::Int8).unwrap();
    let err = dlm::loader::load_generator(&store, &config, 8)
        .err()
        .expect("an int4 pack loaded at int8");
    assert!(err.to_string().contains("made for --quant int4"), "{err}");
    std::fs::remove_dir_all(&out).unwrap();

    let gguf = fixture("ggml_blocks.gguf");
    assert!(dlm::pack::write_pack(&gguf, &out, QuantScheme::Int4, |_, _| {}).is_err());
    assert!(
        dlm::pack::write_pack(&fixture("deepseek_v3"), &out, QuantScheme::Fp16, |_, _| {}).is_err(),
        "fp16 is not a pack precision"
    );
    let _ = std::fs::remove_dir_all(&out);
}
