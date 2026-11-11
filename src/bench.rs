//! `dlm bench`: time prefill and decode on a loaded [`Generator`].
//!
//! The model is built by `serve`'s own loading path, so a number measured here
//! is the number a served request would see with the same flags. A batch decodes
//! through [`Generator::step_sessions`], as the server's scheduler does.

use crate::cli::BenchOpts;
use crate::error::{DlmError, Result};
use crate::forward::ComputeKernel;
use crate::generate::{Generator, Sampler};
use crate::telemetry;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

/// One measured run at one batch size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub batch: usize,
    /// Prompt tokens per second, over every sequence's prefill.
    pub prefill_tok_s: f64,
    /// First sequence: its prefill plus the first decode step (of the whole batch).
    pub ttft_ms: f64,
    /// Generated tokens per second, summed over the batch.
    pub decode_tok_s: f64,
    /// Wall time for one decode step of every sequence in the batch.
    pub ms_per_step: f64,
    /// `--breakdown`: milliseconds per generated token spent in each streaming
    /// stage. Empty when off, or on paths that emit no stage events.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stage_ms_per_token: BTreeMap<String, f64>,
    /// Device memory in use at the end of decode, with every sequence's KV cache
    /// still allocated. Whole device, so other processes count. `None` without
    /// a GPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_used_bytes: Option<u64>,
    /// This process's peak resident memory at the end of the run (a high-water
    /// mark, so it covers every run before it too).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_bytes: Option<u64>,
    /// Speculative runs: the share of proposed tokens the target accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accept_rate: Option<f64>,
}

/// What `--json` writes and `--compare` reads back: the runs, plus what was run
/// and where, so two files can be told apart later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub version: String,
    /// The commit the binary was built from, when the build recorded one
    /// (`DLM_GIT_COMMIT` at build time).
    #[serde(default)]
    pub commit: Option<String>,
    /// Seconds since the Unix epoch.
    #[serde(default)]
    pub timestamp: u64,
    pub model: String,
    pub device: String,
    pub gpu_backend: String,
    #[serde(default)]
    pub gpu_name: Option<String>,
    pub quant: String,
    pub stream: bool,
    pub kv_quant: String,
    /// `greedy`, or the `--temperature`/`--top-p`/`--top-k` the run sampled with.
    #[serde(default)]
    pub sampling: String,
    pub context_length: u32,
    pub prompt_len: usize,
    pub gen_len: usize,
    pub runs: Vec<RunResult>,
    /// The median of each metric per batch size: what `--compare` compares.
    pub summary: Vec<RunResult>,
    pub peak_rss_bytes: Option<u64>,
    pub vram_used_bytes: Option<u64>,
}

/// One metric of one batch size, baseline against this run.
#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    pub batch: usize,
    pub metric: &'static str,
    pub before: f64,
    pub after: f64,
    /// `(after - before) / before`, in percent.
    pub change_pct: f64,
    /// A throughput metric that fell by more than the tolerance.
    pub regressed: bool,
}

/// Compare two summaries batch by batch. Every metric is listed, but only
/// prefill and decode tok/s can regress: latency follows from them, and memory
/// moves for reasons (other processes on the GPU) a speed check should not fail
/// on. Batch sizes measured on only one side are skipped.
pub fn compare(base: &[RunResult], new: &[RunResult], tolerance_pct: f64) -> Vec<Delta> {
    let mut out = Vec::new();
    for n in new {
        let Some(b) = base.iter().find(|b| b.batch == n.batch) else {
            continue;
        };
        let mb = |x: Option<u64>| x.map(|v| v as f64 / (1u64 << 20) as f64);
        let metrics = [
            (
                "prefill tok/s",
                Some(b.prefill_tok_s),
                Some(n.prefill_tok_s),
                true,
            ),
            ("ttft ms", Some(b.ttft_ms), Some(n.ttft_ms), false),
            (
                "decode tok/s",
                Some(b.decode_tok_s),
                Some(n.decode_tok_s),
                true,
            ),
            ("ms/step", Some(b.ms_per_step), Some(n.ms_per_step), false),
            (
                "peak rss MiB",
                mb(b.peak_rss_bytes),
                mb(n.peak_rss_bytes),
                false,
            ),
            (
                "vram MiB",
                mb(b.vram_used_bytes),
                mb(n.vram_used_bytes),
                false,
            ),
        ];
        for (metric, before, after, gated) in metrics {
            let (Some(before), Some(after)) = (before, after) else {
                continue;
            };
            let change_pct = if before == 0.0 {
                0.0
            } else {
                (after - before) / before * 100.0
            };
            out.push(Delta {
                batch: n.batch,
                metric,
                before,
                after,
                change_pct,
                regressed: gated && change_pct < -tolerance_pct,
            });
        }
    }
    out
}

/// A prompt of exactly `len` tokens: `seed` repeated, folded into the vocab.
/// Real text rather than arbitrary ids, so MoE routing and expert-cache hits
/// look like a real request's.
pub fn prompt_ids(seed: &[u32], len: usize, vocab_size: usize) -> Vec<u32> {
    let seed = if seed.is_empty() { &[0u32][..] } else { seed };
    seed.iter()
        .cycle()
        .take(len)
        .map(|&t| t % vocab_size as u32)
        .collect()
}

/// The sampler `opts` asks for, seeded with `seed`: greedy unless
/// `--temperature` is above 0.
pub fn sampler(opts: &BenchOpts, seed: u64) -> Sampler {
    if opts.temperature > 0.0 {
        Sampler::TopPK {
            temperature: opts.temperature,
            top_p: opts.top_p,
            top_k: opts.top_k,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed,
        }
    } else {
        Sampler::Greedy
    }
}

/// Run every batch size in `opts.batch`, `opts.runs` times each, greedily or
/// with the sampler `opts` asks for. With `ngram_gamma`, a batch of one decodes
/// speculatively, proposing that many tokens a round by n-gram lookup, as a
/// lone served request does under `--speculative ngram`; larger batches decode
/// plainly, as the scheduler runs them.
pub fn run<K: ComputeKernel>(
    generator: &Generator<K>,
    prompt: &[u32],
    opts: &BenchOpts,
    ngram_gamma: Option<usize>,
    mut on_result: impl FnMut(&RunResult, usize),
) -> Result<Vec<RunResult>> {
    if prompt.is_empty() || opts.gen_len == 0 || opts.runs == 0 {
        return Err(DlmError::InvalidConfig(
            "bench needs --prompt-len, --gen-len and --runs above 0".into(),
        ));
    }
    if opts.breakdown {
        telemetry::enable();
    }
    let results = (|| {
        let mut results = Vec::new();
        for &batch in &opts.batch {
            for run in 0..opts.runs {
                let r = match ngram_gamma {
                    Some(gamma) if batch <= 1 => {
                        run_speculative(generator, prompt, opts.gen_len, gamma, sampler(opts, 1))?
                    }
                    _ => run_once(
                        generator,
                        prompt,
                        opts.gen_len,
                        batch.max(1),
                        opts.breakdown,
                        |i| sampler(opts, i as u64 + 1),
                    )?,
                };
                on_result(&r, run);
                results.push(r);
            }
        }
        Ok(results)
    })();
    if opts.breakdown {
        telemetry::disable();
    }
    results
}

fn run_once<K: ComputeKernel>(
    generator: &Generator<K>,
    prompt: &[u32],
    gen_len: usize,
    batch: usize,
    breakdown: bool,
    sampler: impl Fn(usize) -> Sampler,
) -> Result<RunResult> {
    let prefill_start = Instant::now();
    let mut first_prefill = 0.0;
    let mut sessions = Vec::with_capacity(batch);
    for i in 0..batch {
        sessions.push(generator.start_session(prompt, sampler(i))?);
        if i == 0 {
            first_prefill = prefill_start.elapsed().as_secs_f64();
        }
    }
    let prefill = prefill_start.elapsed().as_secs_f64();

    // Stage events from prefill are not decode's; start the tally clean.
    let mut stage_us: BTreeMap<String, u64> = BTreeMap::new();
    if breakdown {
        telemetry::drain(usize::MAX);
    }
    let decode_start = Instant::now();
    let mut first_step = 0.0;
    for step in 0..gen_len {
        let t = Instant::now();
        let mut batch_refs: Vec<_> = sessions.iter_mut().collect();
        generator.step_sessions(&mut batch_refs)?;
        if step == 0 {
            first_step = t.elapsed().as_secs_f64();
        }
        // Drain every step so a long run never overflows the ring.
        if breakdown {
            for e in telemetry::drain(usize::MAX).events {
                *stage_us.entry(format!("{:?}", e.stage)).or_default() += e.dur_us as u64;
            }
        }
    }
    let decode = decode_start.elapsed().as_secs_f64();
    // Before `sessions` drops: their KV buffers are freed with them.
    let vram_used_bytes = crate::gpu::mem_get_info().ok().map(|m| m.total - m.free);

    let tokens = (batch * gen_len) as f64;
    Ok(RunResult {
        batch,
        prefill_tok_s: (batch * prompt.len()) as f64 / prefill,
        ttft_ms: (first_prefill + first_step) * 1e3,
        decode_tok_s: tokens / decode,
        ms_per_step: decode * 1e3 / gen_len as f64,
        stage_ms_per_token: stage_us
            .into_iter()
            .map(|(k, us)| (k, us as f64 / 1e3 / tokens))
            .collect(),
        vram_used_bytes,
        peak_rss_bytes: peak_rss_bytes(),
        accept_rate: None,
    })
}

/// One sequence decoded speculatively by n-gram lookup ([`SpeculativeSession`]):
/// rounds until `gen_len` tokens are out. `ms_per_step` is per token, and
/// `accept_rate` the share of proposals the target kept.
///
/// [`SpeculativeSession`]: crate::speculative::SpeculativeSession
fn run_speculative<K: ComputeKernel>(
    generator: &Generator<K>,
    prompt: &[u32],
    gen_len: usize,
    gamma: usize,
    sampler: Sampler,
) -> Result<RunResult> {
    let prefill_start = Instant::now();
    let mut session =
        crate::speculative::SpeculativeSession::ngram(generator, gamma, prompt, sampler)?;
    let prefill = prefill_start.elapsed().as_secs_f64();
    let decode_start = Instant::now();
    let mut first_round = 0.0;
    let mut produced = 0;
    while produced < gen_len {
        produced += session.step(gen_len - produced)?.len();
        if first_round == 0.0 {
            first_round = decode_start.elapsed().as_secs_f64();
        }
    }
    let decode = decode_start.elapsed().as_secs_f64();
    let vram_used_bytes = crate::gpu::mem_get_info().ok().map(|m| m.total - m.free);
    Ok(RunResult {
        batch: 1,
        prefill_tok_s: prompt.len() as f64 / prefill,
        ttft_ms: (prefill + first_round) * 1e3,
        decode_tok_s: gen_len as f64 / decode,
        ms_per_step: decode * 1e3 / gen_len as f64,
        stage_ms_per_token: BTreeMap::new(),
        vram_used_bytes,
        peak_rss_bytes: peak_rss_bytes(),
        accept_rate: Some(if session.proposed() == 0 {
            0.0
        } else {
            session.accepted() as f64 / session.proposed() as f64
        }),
    })
}

/// The median of each metric across runs, one row per batch size.
pub fn summarize(results: &[RunResult]) -> Vec<RunResult> {
    let mut batches: Vec<usize> = Vec::new();
    for r in results {
        if !batches.contains(&r.batch) {
            batches.push(r.batch);
        }
    }
    batches
        .into_iter()
        .map(|batch| {
            let runs: Vec<&RunResult> = results.iter().filter(|r| r.batch == batch).collect();
            let med = |f: &dyn Fn(&RunResult) -> f64| {
                let mut v: Vec<f64> = runs.iter().map(|r| f(r)).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let mut stages = BTreeMap::new();
            for k in runs.iter().flat_map(|r| r.stage_ms_per_token.keys()) {
                let k = k.clone();
                let m = med(&|r| r.stage_ms_per_token.get(&k).copied().unwrap_or(0.0));
                stages.insert(k, m);
            }
            RunResult {
                batch,
                prefill_tok_s: med(&|r| r.prefill_tok_s),
                ttft_ms: med(&|r| r.ttft_ms),
                decode_tok_s: med(&|r| r.decode_tok_s),
                ms_per_step: med(&|r| r.ms_per_step),
                stage_ms_per_token: stages,
                vram_used_bytes: runs.iter().filter_map(|r| r.vram_used_bytes).max(),
                peak_rss_bytes: runs.iter().filter_map(|r| r.peak_rss_bytes).max(),
                accept_rate: runs[0]
                    .accept_rate
                    .map(|_| med(&|r| r.accept_rate.unwrap_or(0.0))),
            }
        })
        .collect()
}

/// Peak resident memory of this process so far, where the platform reports it.
#[cfg(unix)]
pub fn peak_rss_bytes() -> Option<u64> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return None;
    }
    let max = usage.ru_maxrss as u64;
    // Linux reports kilobytes, macOS bytes.
    Some(if cfg!(target_os = "macos") {
        max
    } else {
        max * 1024
    })
}

/// Peak resident memory of this process so far, where the platform reports it.
#[cfg(windows)]
pub fn peak_rss_bytes() -> Option<u64> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    let mut c: ProcessMemoryCounters = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (ok != 0).then_some(c.peak_working_set_size as u64)
}

/// Peak resident memory of this process so far, where the platform reports it.
#[cfg(not(any(unix, windows)))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(batch: usize, decode: f64) -> RunResult {
        RunResult {
            batch,
            prefill_tok_s: decode * 10.0,
            ttft_ms: 1.0,
            decode_tok_s: decode,
            ms_per_step: 1e3 / decode,
            stage_ms_per_token: BTreeMap::new(),
            vram_used_bytes: None,
            peak_rss_bytes: Some(1 << 30),
            accept_rate: None,
        }
    }

    fn report(summary: Vec<RunResult>) -> Report {
        Report {
            version: "0.4.0".into(),
            commit: Some("abc1234".into()),
            timestamp: 1_762_700_000,
            model: "models/Qwen2.5-0.5B-Instruct".into(),
            device: "gpu".into(),
            gpu_backend: "CUDA".into(),
            gpu_name: Some("NVIDIA GeForce GTX 1650".into()),
            quant: "Int4".into(),
            stream: false,
            kv_quant: "f16".into(),
            sampling: "greedy".into(),
            context_length: 1024,
            prompt_len: 512,
            gen_len: 64,
            runs: summary.clone(),
            summary,
            peak_rss_bytes: Some(1 << 30),
            vram_used_bytes: None,
        }
    }

    #[test]
    fn report_round_trips_through_json() {
        let mut r = row(4, 35.1);
        r.stage_ms_per_token.insert("H2D".into(), 1.5);
        r.vram_used_bytes = Some(1 << 31);
        let before = report(vec![row(1, 38.8), r]);
        let text = serde_json::to_string_pretty(&before).unwrap();
        let after: Report = serde_json::from_str(&text).unwrap();
        assert_eq!(serde_json::to_string_pretty(&after).unwrap(), text);
        assert_eq!(after.summary[1].stage_ms_per_token["H2D"], 1.5);
    }

    #[test]
    fn reads_a_report_written_before_the_new_fields() {
        // What `--json` wrote before it recorded commit, time, GPU and per-row RSS.
        let old = r#"{"version":"0.4.0","model":"m","device":"cpu","gpu_backend":"none",
            "quant":"Fp16","stream":false,"kv_quant":"f16","context_length":512,
            "prompt_len":128,"gen_len":16,"runs":[],"summary":[{"batch":1,
            "prefill_tok_s":9.1,"ttft_ms":14233.0,"decode_tok_s":7.38,"ms_per_step":135.5}],
            "peak_rss_bytes":2867000000,"vram_used_bytes":null}"#;
        let r: Report = serde_json::from_str(old).unwrap();
        assert_eq!((r.commit, r.summary[0].peak_rss_bytes), (None, None));
        assert_eq!(r.summary[0].decode_tok_s, 7.38);
    }

    #[test]
    fn compare_flags_only_throughput_beyond_tolerance() {
        let base = [row(1, 40.0), row(4, 100.0)];
        let mut slow = row(1, 37.0); // -7.5% decode, -7.5% prefill
        slow.ttft_ms = 10.0; // 10x worse latency, but latency never fails the check
        let new = [slow, row(4, 98.0), row(8, 1.0)]; // -2%; batch 8 has no baseline
        let d = compare(&base, &new, 5.0);
        let regressed: Vec<_> = d
            .iter()
            .filter(|d| d.regressed)
            .map(|d| (d.batch, d.metric))
            .collect();
        assert_eq!(regressed, [(1, "prefill tok/s"), (1, "decode tok/s")]);
        let decode = d
            .iter()
            .find(|d| d.batch == 1 && d.metric == "decode tok/s")
            .unwrap();
        assert!((decode.change_pct + 7.5).abs() < 1e-9);
        assert!(d.iter().all(|d| d.batch != 8));
        // A looser tolerance lets the same run pass, and a faster one always does.
        assert!(compare(&base, &new, 10.0).iter().all(|d| !d.regressed));
        assert!(compare(&base, &[row(1, 80.0)], 0.0)
            .iter()
            .all(|d| !d.regressed));
        // Memory only shows when both sides measured it.
        assert!(d.iter().any(|d| d.metric == "peak rss MiB"));
        assert!(d.iter().all(|d| d.metric != "vram MiB"));
    }

    #[test]
    fn summary_takes_the_median_per_batch() {
        let rows = [row(1, 9.0), row(1, 1.0), row(1, 5.0), row(4, 20.0)];
        let s = summarize(&rows);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].batch, s[0].decode_tok_s), (1, 5.0));
        assert_eq!((s[1].batch, s[1].decode_tok_s), (4, 20.0));
    }

    #[test]
    fn prompt_is_exact_length_and_in_vocab() {
        let p = prompt_ids(&[3, 900, 7], 8, 10);
        assert_eq!(p, vec![3, 0, 7, 3, 0, 7, 3, 0]);
        assert_eq!(prompt_ids(&[], 2, 10), vec![0, 0]);
    }

    #[test]
    fn peak_rss_is_reported_on_desktop_platforms() {
        if cfg!(any(unix, windows)) {
            assert!(peak_rss_bytes().unwrap() > 0);
        }
    }
}
