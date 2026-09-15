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
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Instant;

/// One measured run at one batch size.
#[derive(Debug, Clone, Serialize)]
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
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub stage_ms_per_token: BTreeMap<String, f64>,
    /// Device memory in use at the end of decode, with every sequence's KV cache
    /// still allocated. Whole device, so other processes count. `None` without
    /// a GPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_used_bytes: Option<u64>,
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

/// Run every batch size in `opts.batch`, `opts.runs` times each, greedily.
pub fn run<K: ComputeKernel>(
    generator: &Generator<K>,
    prompt: &[u32],
    opts: &BenchOpts,
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
                let r = run_once(
                    generator,
                    prompt,
                    opts.gen_len,
                    batch.max(1),
                    opts.breakdown,
                )?;
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
) -> Result<RunResult> {
    let prefill_start = Instant::now();
    let mut first_prefill = 0.0;
    let mut sessions = Vec::with_capacity(batch);
    for i in 0..batch {
        sessions.push(generator.start_session(prompt, Sampler::Greedy)?);
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
        }
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
