//! Continuous batching (`PRD.md` §3.3).
//!
//! Instead of running requests one after another, the scheduler keeps up to
//! `max_batch` generations **in flight at once** and advances every active one
//! by a single token per tick, admitting queued requests into slots as they free
//! up (rather than waiting for a whole batch to finish). This is the
//! "continuous" / in-flight batching that keeps the engine busy under a stream
//! of requests.
//!
//! Each request runs in its own [`GenerationSession`] with independent KV state,
//! so interleaving is transparent: a request's output is identical to running it
//! alone. On a batched forward kernel the per-tick step would fuse all active
//! sequences into one matmul; here it steps them in a loop — same scheduling,
//! same output, the fused speedup awaiting a batch kernel.
//!
//! ## Speculative decoding
//!
//! When constructed with [`with_speculative`](BatchScheduler::with_speculative),
//! each slot instead runs a [`SpeculativeSession`]: a tick advances a request by
//! one *round* rather than one token, emitting the 1..=`gamma`+1 tokens that
//! round accepted. The output is still exactly target-greedy — identical to the
//! plain path — and every emitted token flows through the same `produced` list,
//! so streaming and length/EOS handling are unchanged. A slot with two draft
//! rejections in a row simply emits one token per round, matching plain decoding.

use crate::error::Result;
use crate::forward::{ComputeKernel, KvSnapshot};
use crate::generate::{GenerationSession, Generator, Sampler};
use crate::speculative::SpeculativeSession;
use std::collections::VecDeque;

/// A bounded cache of KV snapshots keyed by the prompt tokens that produced
/// them, so a request whose prompt extends a cached one can resume from the
/// snapshot instead of re-prefilling the shared prefix (a big win for chat
/// traffic sharing a system prompt). Off unless enabled; only used on the plain
/// (non-speculative) path, which is the only one that can resume.
struct PrefixCache {
    max_entries: usize,
    /// `(prompt tokens, snapshot after prefilling them)`, oldest first.
    entries: Vec<(Vec<u32>, KvSnapshot)>,
}

impl PrefixCache {
    fn new(max_entries: usize) -> Self {
        Self { max_entries, entries: Vec::new() }
    }

    /// The snapshot of the longest cached prompt that is a strict prefix of
    /// `prompt` (so there is a non-empty suffix left to prefill).
    // Linear scan, deliberately: the cache is hard-bounded by `max_entries` (the
    // --prefix-cache-size flag) in `insert`, so this is O(configured size), not
    // O(traffic). Index by first token only if that bound is ever raised to
    // something large. The bound is enforced by
    // `prefix_cache_never_exceeds_its_bound`.
    fn longest_prefix(&self, prompt: &[u32]) -> Option<KvSnapshot> {
        self.entries
            .iter()
            .filter(|(toks, _)| toks.len() < prompt.len() && prompt.starts_with(toks))
            .max_by_key(|(toks, _)| toks.len())
            .map(|(_, snap)| snap.clone())
    }

    /// Cache `prompt`'s post-prefill `snapshot`, evicting the oldest entry when
    /// full. Replaces any existing entry for the same prompt.
    fn insert(&mut self, prompt: Vec<u32>, snapshot: KvSnapshot) {
        if self.max_entries == 0 {
            return;
        }
        self.entries.retain(|(toks, _)| toks != &prompt);
        self.entries.push((prompt, snapshot));
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }
}

/// A queued request awaiting admission.
struct Pending {
    id: u64,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    /// Stop when any of these token ids is produced; empty means run to length.
    eos: Vec<u32>,
    /// Sampler for the plain (non-speculative) path. Ignored when the scheduler
    /// speculates — speculative decoding is greedy-exact by construction.
    sampler: Sampler,
}

/// The decoder backing one in-flight slot: plain single-token stepping, or a
/// speculative session that emits a whole round's accepted tokens per step.
enum Decoder<'a, K: ComputeKernel> {
    Plain(GenerationSession<'a, K>),
    Speculative(SpeculativeSession<'a, K, K>),
}

/// An in-flight generation occupying a batch slot.
struct Active<'a, K: ComputeKernel> {
    id: u64,
    decoder: Decoder<'a, K>,
    remaining: usize,
    eos: Vec<u32>,
}

/// A completed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finished {
    pub id: u64,
    pub tokens: Vec<u32>,
}

/// Speculative acceptance for a single request: how many draft tokens it
/// proposed and how many the target accepted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AcceptanceStats {
    pub proposed: usize,
    pub accepted: usize,
}

impl AcceptanceStats {
    /// Fraction of proposed draft tokens accepted (0.0–1.0; 0.0 if none proposed).
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }
}

/// What one scheduler tick produced: `(request id, token)` pairs emitted this
/// step, and the ids of requests that finished (their last token is in
/// `produced`). Streaming consumers forward `produced` and close on `finished`.
///
/// `finished_stats` carries the speculative acceptance for each finished request
/// that was decoding speculatively; it is empty on the plain path.
#[derive(Debug, Clone, Default)]
pub struct Tick {
    pub produced: Vec<(u64, u32)>,
    pub finished: Vec<u64>,
    pub finished_stats: Vec<(u64, AcceptanceStats)>,
}

/// A continuous-batching scheduler over a borrowed generator.
pub struct BatchScheduler<'a, K: ComputeKernel> {
    generator: &'a Generator<K>,
    /// Optional draft model; when present every slot decodes speculatively.
    draft: Option<&'a Generator<K>>,
    /// Draft tokens proposed per round when speculating.
    gamma: usize,
    max_batch: usize,
    pending: VecDeque<Pending>,
    active: Vec<Active<'a, K>>,
    /// Cumulative draft tokens proposed / accepted across retired slots.
    proposed: usize,
    accepted: usize,
    /// Optional cross-request prefix cache (plain path only).
    prefix_cache: Option<PrefixCache>,
    /// Count of admissions that resumed from a cached prefix (cache hits).
    resume_hits: usize,
}

impl<'a, K: ComputeKernel> BatchScheduler<'a, K> {
    /// Create a scheduler running at most `max_batch` concurrent generations.
    pub fn new(generator: &'a Generator<K>, max_batch: usize) -> Self {
        Self {
            generator,
            draft: None,
            gamma: 0,
            max_batch: max_batch.max(1),
            pending: VecDeque::new(),
            active: Vec::new(),
            proposed: 0,
            accepted: 0,
            prefix_cache: None,
            resume_hits: 0,
        }
    }

    /// Enable a cross-request prefix cache holding up to `max_entries` KV
    /// snapshots. A request whose prompt extends a cached one resumes from the
    /// snapshot, skipping the shared prefix's prefill. No effect on a
    /// speculative scheduler (its sessions can't resume). `0` disables it.
    pub fn with_prefix_cache(mut self, max_entries: usize) -> Self {
        self.prefix_cache = if max_entries > 0 && self.draft.is_none() {
            Some(PrefixCache::new(max_entries))
        } else {
            None
        };
        self
    }

    /// Create a scheduler that decodes every request speculatively, using
    /// `draft` to propose `gamma` tokens per round for the `generator` (target)
    /// to verify. Output is identical to [`new`](Self::new); only faster when the
    /// draft guesses well. `draft` must share the target's tokenizer/vocabulary.
    pub fn with_speculative(
        generator: &'a Generator<K>,
        draft: &'a Generator<K>,
        max_batch: usize,
        gamma: usize,
    ) -> Self {
        Self {
            generator,
            draft: Some(draft),
            gamma: gamma.max(1),
            max_batch: max_batch.max(1),
            pending: VecDeque::new(),
            active: Vec::new(),
            proposed: 0,
            accepted: 0,
            prefix_cache: None,
            resume_hits: 0,
        }
    }

    /// Cumulative `(proposed, accepted)` draft-token counts over all *retired*
    /// speculative slots. Zero unless the scheduler was built with a draft.
    pub fn speculative_stats(&self) -> (usize, usize) {
        (self.proposed, self.accepted)
    }

    /// How many admitted requests resumed from a cached prefix (prefix-cache
    /// hits). Zero unless [`with_prefix_cache`](Self::with_prefix_cache) is on.
    pub fn resume_hits(&self) -> usize {
        self.resume_hits
    }

    /// Queue a request (greedy decoding). Errors on an empty prompt. `eos` lists
    /// the token ids that stop generation (empty = run to length).
    pub fn submit(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        eos: Vec<u32>,
    ) -> Result<()> {
        self.submit_sampled(id, prompt, max_new_tokens, eos, Sampler::Greedy)
    }

    /// Queue a request with an explicit `sampler` for the plain path. The sampler
    /// is honored only when the scheduler is not speculating (speculative
    /// decoding is greedy-exact); otherwise it is ignored.
    pub fn submit_sampled(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        eos: Vec<u32>,
        sampler: Sampler,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(crate::error::DlmError::InvalidConfig("prompt is empty".into()));
        }
        self.pending.push_back(Pending {
            id,
            prompt,
            max_new_tokens,
            eos,
            sampler,
        });
        Ok(())
    }

    /// Abandon a request (e.g. the client disconnected): drop it from the
    /// pending queue and retire its in-flight slot, freeing its KV state. No-op
    /// if the id is unknown. Returns true if anything was removed.
    pub fn abort(&mut self, id: u64) -> bool {
        let before = self.pending.len() + self.active.len();
        self.pending.retain(|p| p.id != id);
        self.active.retain(|a| a.id != id);
        before != self.pending.len() + self.active.len()
    }

    /// Whether any request is pending or in flight.
    pub fn has_work(&self) -> bool {
        !self.pending.is_empty() || !self.active.is_empty()
    }

    /// In-flight request count.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// Fill free slots from the pending queue (prefilling each new session).
    /// Returns ids of requests that finished immediately (zero max tokens).
    fn admit(&mut self) -> Result<Vec<u64>> {
        let mut zero_finished = Vec::new();
        while self.active.len() < self.max_batch {
            let Some(p) = self.pending.pop_front() else { break };
            if p.max_new_tokens == 0 {
                zero_finished.push(p.id);
                continue;
            }
            let decoder = match self.draft {
                Some(draft) => Decoder::Speculative(SpeculativeSession::new(
                    self.generator,
                    draft,
                    self.gamma,
                    &p.prompt,
                )),
                None => {
                    // Resume from the longest cached prefix if one exists;
                    // otherwise prefill from scratch. Either way, cache the
                    // full-prompt snapshot so later requests can extend it.
                    let resume = self
                        .prefix_cache
                        .as_ref()
                        .and_then(|c| c.longest_prefix(&p.prompt));
                    let mut session = match resume {
                        Some(snap) => {
                            self.resume_hits += 1;
                            // Pass the whole prompt so the repetition penalty covers
                            // the cached prefix too (resume derives the suffix).
                            self.generator.resume_session(snap, &p.prompt, p.sampler)?
                        }
                        None => self.generator.start_session(&p.prompt, p.sampler)?,
                    };
                    if let Some(cache) = self.prefix_cache.as_mut() {
                        // `_synced`: on the GPU kernels the real K/V is in VRAM,
                        // so an unsynced snapshot would cache zeros.
                        cache.insert(p.prompt.clone(), session.snapshot_synced()?);
                    }
                    Decoder::Plain(session)
                }
            };
            self.active.push(Active {
                id: p.id,
                decoder,
                remaining: p.max_new_tokens,
                eos: p.eos,
            });
        }
        Ok(zero_finished)
    }

    /// One scheduler tick: admit queued requests, advance every active slot (by
    /// one token normally, or one accept/reject round when speculating), and
    /// retire any that finished. Returns the tokens produced and the ids that
    /// completed this tick (for streaming).
    pub fn step(&mut self) -> Result<Tick> {
        let zero_finished = self.admit()?;
        let mut tick = Tick {
            finished: zero_finished,
            ..Default::default()
        };
        let mut still_active = Vec::with_capacity(self.active.len());
        for mut a in self.active.drain(..) {
            // Plain slots yield one token; speculative slots yield a whole
            // round's accepted tokens (never more than `remaining`).
            let emitted = match &mut a.decoder {
                Decoder::Plain(s) => vec![s.step()?],
                Decoder::Speculative(s) => s.step(a.remaining)?,
            };

            // Emit tokens in order, stopping the request at the length cap or an
            // EOS (inclusive) — any tokens past that point in the round are
            // discarded, exactly as plain decoding would have stopped there.
            let mut done = false;
            for token in emitted {
                tick.produced.push((a.id, token));
                a.remaining -= 1;
                if a.remaining == 0 || a.eos.contains(&token) {
                    done = true;
                    break;
                }
            }

            if done {
                if let Decoder::Speculative(s) = &a.decoder {
                    let stats = AcceptanceStats {
                        proposed: s.proposed(),
                        accepted: s.accepted(),
                    };
                    self.proposed += stats.proposed;
                    self.accepted += stats.accepted;
                    tick.finished_stats.push((a.id, stats));
                }
                tick.finished.push(a.id);
            } else {
                still_active.push(a);
            }
        }
        self.active = still_active;
        Ok(tick)
    }

    /// Run ticks until every request has completed, returning the results (in
    /// completion order). Convenience wrapper over [`step`](Self::step).
    pub fn run(&mut self) -> Result<Vec<Finished>> {
        use std::collections::HashMap;
        let mut outputs: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut results = Vec::new();
        while self.has_work() {
            let tick = self.step()?;
            for (id, token) in tick.produced {
                outputs.entry(id).or_default().push(token);
            }
            for id in tick.finished {
                results.push(Finished {
                    id,
                    tokens: outputs.remove(&id).unwrap_or_default(),
                });
            }
        }
        Ok(results)
    }
}


#[cfg(test)]
mod prefix_cache_tests {
    use super::PrefixCache;
    use crate::cache::{KvCacheConfig, PagedKvCache};
    use crate::forward::{ForwardOrchestrator, KvSnapshot, StubKernel};

    /// A throwaway snapshot; these tests exercise the cache's bookkeeping, not
    /// the KV contents.
    fn snap() -> KvSnapshot {
        let kv_cfg = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            block_size: 16,
        };
        let orch = ForwardOrchestrator::new(
            StubKernel::new(1, 4, 2),
            PagedKvCache::new(kv_cfg, 4),
            crate::forward::KvQuant::None,
        );
        orch.snapshot()
    }

    /// `longest_prefix` scans linearly, which is only acceptable because the
    /// cache is bounded. Pin the bound so that stays true: inserting far more
    /// entries than the limit must evict, never grow.
    #[test]
    fn prefix_cache_never_exceeds_its_bound() {
        let mut cache = PrefixCache::new(3);
        for i in 0..50u32 {
            cache.insert(vec![i, i + 1], snap());
            assert!(
                cache.entries.len() <= 3,
                "cache grew to {} past its bound of 3",
                cache.entries.len()
            );
        }
        assert_eq!(cache.entries.len(), 3);

        // Size 0 disables it entirely rather than caching one entry.
        let mut off = PrefixCache::new(0);
        off.insert(vec![1, 2], snap());
        assert!(off.entries.is_empty());
    }

    /// Re-inserting a prompt replaces its entry instead of duplicating it —
    /// otherwise a hot prompt would evict everything else out of the bound.
    #[test]
    fn prefix_cache_replaces_rather_than_duplicates() {
        let mut cache = PrefixCache::new(4);
        for _ in 0..5 {
            cache.insert(vec![7, 8, 9], snap());
        }
        assert_eq!(cache.entries.len(), 1);
    }

    /// The match must be a *strict* prefix (a non-empty suffix left to prefill),
    /// and the longest one available. An exact-length match would leave nothing
    /// to decode from.
    #[test]
    fn longest_prefix_picks_the_longest_strict_prefix() {
        let mut cache = PrefixCache::new(8);
        cache.insert(vec![1], snap());
        cache.insert(vec![1, 2, 3], snap());
        cache.insert(vec![1, 2], snap());
        cache.insert(vec![9, 9], snap());

        // Longest strict prefix of [1,2,3,4] is [1,2,3].
        assert!(cache.longest_prefix(&[1, 2, 3, 4]).is_some());
        // Exact match is not a *strict* prefix — nothing would be left to prefill.
        assert!(cache.longest_prefix(&[1]).is_none());
        // No shared prefix at all.
        assert!(cache.longest_prefix(&[5, 5, 5]).is_none());
    }
}
