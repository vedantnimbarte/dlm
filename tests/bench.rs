//! `dlm bench` end to end: the binary loads a tiny checkpoint through `serve`'s
//! path, measures it, and writes a JSON report with real numbers in it. No
//! timing assertions: CI machines vary too much for a speed gate to mean
//! anything, so this only proves the harness runs and reports.

use std::io::Write;
use std::process::Command;

/// Write an F32 safetensors checkpoint from named 1-D tensors.
fn write_f32_model(dir: &std::path::Path, tensors: &[(String, Vec<f32>)]) {
    let mut entries = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut offset = 0usize;
    for (name, values) in tensors {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[{offset},{}]}}"#,
            values.len(),
            offset + bytes.len()
        ));
        data.extend_from_slice(&bytes);
        offset += bytes.len();
    }
    let header = format!("{{{}}}", entries.join(","));
    let path = dir.join("model-00001-of-00001.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&data).unwrap();
}

/// A 4-layer Llama-shaped checkpoint with a 300-token vocab, so the byte-level
/// fallback tokenizer's ids fit without folding.
fn write_checkpoint(dir: &std::path::Path) {
    let (h, nh, nkv, hd, inter, vocab, layers) =
        (8usize, 2usize, 1usize, 4usize, 16usize, 300usize, 4);
    let fill = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i + seed) % 13) as f32 - 6.0) * 0.01)
            .collect()
    };
    let mut t: Vec<(String, Vec<f32>)> =
        vec![("model.embed_tokens.weight".into(), fill(1, vocab * h))];
    for i in 0..layers {
        let p = format!("model.layers.{i}.");
        t.push((format!("{p}self_attn.q_proj.weight"), fill(i, nh * hd * h)));
        t.push((format!("{p}self_attn.k_proj.weight"), fill(i, nkv * hd * h)));
        t.push((format!("{p}self_attn.v_proj.weight"), fill(i, nkv * hd * h)));
        t.push((format!("{p}self_attn.o_proj.weight"), fill(i, h * nh * hd)));
        t.push((format!("{p}mlp.gate_proj.weight"), fill(i, inter * h)));
        t.push((format!("{p}mlp.up_proj.weight"), fill(i, inter * h)));
        t.push((format!("{p}mlp.down_proj.weight"), fill(i, h * inter)));
        t.push((format!("{p}input_layernorm.weight"), vec![1.0; h]));
        t.push((format!("{p}post_attention_layernorm.weight"), vec![1.0; h]));
    }
    t.push(("model.norm.weight".into(), vec![1.0; h]));
    t.push(("lm_head.weight".into(), fill(3, vocab * h)));
    write_f32_model(dir, &t);
    std::fs::write(
        dir.join("config.json"),
        format!(
            r#"{{"hidden_size":{h},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"num_hidden_layers":{layers},"vocab_size":{vocab},"intermediate_size":{inter}}}"#
        ),
    )
    .unwrap();
}

fn bench(dir: &std::path::Path, extra: &[&str]) -> serde_json::Value {
    let json = dir.join(format!("bench-{}.json", extra.len()));
    let out = Command::new(env!("CARGO_BIN_EXE_dlm"))
        .arg("bench")
        .arg("--model-path")
        .arg(dir)
        .args(["--device", "cpu", "--context-length", "64"])
        .args([
            "--prompt-len",
            "20",
            "--gen-len",
            "6",
            "--batch",
            "1,2",
            "--runs",
            "2",
        ])
        .arg("--json")
        .arg(&json)
        .args(extra)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "bench failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap()
}

#[test]
fn bench_reports_every_run_and_a_median_per_batch() {
    let tmp = tempfile::tempdir().unwrap();
    write_checkpoint(tmp.path());
    let report = bench(tmp.path(), &[]);

    assert_eq!(report["runs"].as_array().unwrap().len(), 4);
    let summary = report["summary"].as_array().unwrap();
    assert_eq!(summary.len(), 2);
    for (row, batch) in summary.iter().zip([1, 2]) {
        assert_eq!(row["batch"], batch);
        for key in ["prefill_tok_s", "ttft_ms", "decode_tok_s", "ms_per_step"] {
            let v = row[key].as_f64().unwrap();
            assert!(v.is_finite() && v > 0.0, "{key} = {v}");
        }
    }
    assert_eq!(report["prompt_len"], 20);
}

#[test]
fn bench_breaks_streamed_decode_down_by_stage() {
    let tmp = tempfile::tempdir().unwrap();
    write_checkpoint(tmp.path());
    // Two resident layers of four and no RAM cache, so decode really streams.
    let report = bench(
        tmp.path(),
        &[
            "--stream",
            "--resident-layers",
            "2",
            "--ram-cache-gb",
            "0",
            "--breakdown",
        ],
    );
    let stages = report["summary"][0]["stage_ms_per_token"]
        .as_object()
        .expect("streamed decode emits stage events");
    assert!(stages.contains_key("Compute"), "stages: {stages:?}");
}

#[test]
fn bench_refuses_a_workload_longer_than_the_context() {
    let tmp = tempfile::tempdir().unwrap();
    write_checkpoint(tmp.path());
    let out = Command::new(env!("CARGO_BIN_EXE_dlm"))
        .arg("bench")
        .arg("--model-path")
        .arg(tmp.path())
        .args(["--context-length", "16", "--prompt-len", "20"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("exceeds --context-length"));
}
