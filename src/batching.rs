//! Continuous batching.
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
//! alone. A tick advances every plain slot through one
//! [`Generator::step_sessions`] call, which a batching kernel (the resident GPU
//! kernel) runs as one fused pass per layer rather than one pass per request.
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
use std::time::{Duration, Instant};

/// How long one tick may spend feeding prompts in while other requests are
/// decoding. A long prompt used to be prefilled whole on admission, stalling
/// every running stream for its full duration (14 s for 1,590 tokens of
/// Qwen2.5-0.5B on a GTX 1650). Now it goes in 16 tokens at a time between
/// decode steps.
const PREFILL_TICK_BUDGET: Duration = Duration::from_millis(200);

/// Prompt tokens fed in per piece: the GPU kernels' prefill group size, so
/// splitting a prompt leaves the arithmetic exactly as one call would.
const PREFILL_PIECE: usize = 16;

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
        Self {
            max_entries,
            entries: Vec::new(),
        }
    }

    /// A snapshot of the longest run of leading tokens `prompt` shares with a
    /// cached prompt, cut down to a multiple of `align` and short of the whole
    /// prompt (a non-empty suffix is left to prefill).
    ///
    /// Shared, not identical: requests with one system prompt and different
    /// questions share the system prompt, and a cached prompt that includes its
    /// own question must still serve the next one. `align` is the KV block size
    /// on a kernel that shares device blocks (only whole blocks can be shared)
    /// and 1 elsewhere.
    // Linear scan, deliberately: the cache is hard-bounded by `max_entries` (the
    // --prefix-cache-size flag) in `insert`, so this is O(configured size), not
    // O(traffic). Index by first token only if that bound is ever raised to
    // something large. The bound is enforced by
    // `prefix_cache_never_exceeds_its_bound`.
    fn longest_prefix(&self, prompt: &[u32], align: usize) -> Option<KvSnapshot> {
        let usable = |toks: &[u32], snap: &KvSnapshot| {
            let common = toks.iter().zip(prompt).take_while(|(a, b)| a == b).count();
            let rows = common.min(prompt.len() - 1).min(snap.position());
            rows / align * align
        };
        self.entries
            .iter()
            .map(|(toks, snap)| (usable(toks, snap), snap))
            .filter(|&(rows, _)| rows > 0)
            .max_by_key(|&(rows, _)| rows)
            .map(|(rows, snap)| snap.prefix(rows))
    }

    /// Drop the oldest entry, releasing whatever KV it holds (on a GPU kernel,
    /// the blocks no running session shares). False if there was none.
    fn evict_oldest(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        self.entries.remove(0);
        true
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
    /// How tokens are chosen, on the plain and the speculative path alike.
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
    /// KV tokens set aside for this request at admission (whole blocks).
    reserved: usize,
    /// The prompt, kept while it is still being prefilled so its prefix can be
    /// cached once it is in. `None` when there is nothing to cache.
    cache_prompt: Option<Vec<u32>>,
}

impl<K: ComputeKernel> Active<'_, K> {
    /// Reserved KV tokens not yet backed by blocks. Blocks are taken a whole
    /// block at a time as rows are written, so a partly filled block counts as
    /// taken.
    fn unallocated(&self) -> usize {
        let rows = match &self.decoder {
            Decoder::Plain(s) => s.kv_rows(),
            Decoder::Speculative(s) => s.kv_rows(),
        };
        self.reserved.saturating_sub(round_to_block(rows))
    }
}

/// Rows per paged KV block. `src/forward/kv_pool.rs` checks that it matches.
pub(crate) const KV_BLOCK_TOKENS: usize = 16;

/// `tokens` rounded up to whole KV blocks, the unit a GPU kernel's pool hands out.
fn round_to_block(tokens: usize) -> usize {
    tokens.div_ceil(KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS
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

    /// Queue a request with an explicit `sampler`. A speculating scheduler
    /// honors it too: its output follows the target's sampled distribution, and
    /// is identical to plain decoding under greedy.
    pub fn submit_sampled(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        eos: Vec<u32>,
        sampler: Sampler,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(crate::error::DlmError::InvalidConfig(
                "prompt is empty".into(),
            ));
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

    /// KV tokens a request needs set aside before it starts: its prompt and every
    /// token it may generate, plus, when speculating, the draft tokens a round can
    /// write before the target rejects them.
    fn reservation(&self, p: &Pending) -> usize {
        let slack = if self.draft.is_some() {
            self.gamma + 1
        } else {
            0
        };
        round_to_block(p.prompt.len() + p.max_new_tokens + slack)
    }

    /// Whether the KV pools can take `need` more tokens on top of what running
    /// requests have reserved but not yet allocated. Kernels without a fixed pool
    /// (CPU) always can. With nothing running a request is always admitted: if it
    /// cannot fit an empty pool, waiting would never make it fit, and it should
    /// fail with the pool's error rather than stall the queue.
    fn kv_fits(&self, need: usize) -> bool {
        let cached = self
            .prefix_cache
            .as_ref()
            .is_some_and(|c| !c.entries.is_empty());
        if self.active.is_empty() && !cached {
            return true;
        }
        let outstanding: usize = self.active.iter().map(Active::unallocated).sum();
        let fits = |free: Option<usize>| free.is_none_or(|free| free >= outstanding + need);
        fits(self.generator.kv_free_tokens()) && fits(self.draft.and_then(|d| d.kv_free_tokens()))
    }

    /// Whether `prompt` starts with at least one KV block of tokens that a request
    /// still being prefilled also starts with. Such a request waits: once the
    /// other prompt is in, its prefix is cached and this one resumes from it,
    /// instead of both computing the shared part. A burst of requests with one
    /// system prompt then computes it once.
    fn shares_a_prompt_in_flight(&self, prompt: &[u32]) -> bool {
        if self.prefix_cache.is_none() {
            return false;
        }
        let align = if self.generator.kv_free_tokens().is_some() {
            KV_BLOCK_TOKENS
        } else {
            1
        };
        self.active.iter().any(|a| {
            a.cache_prompt.as_ref().is_some_and(|other| {
                let common = other.iter().zip(prompt).take_while(|(x, y)| x == y).count();
                common.min(prompt.len() - 1) >= align
            })
        })
    }

    /// Fill free slots from the pending queue (prefilling each new session).
    /// Returns ids of requests that finished immediately (zero max tokens).
    ///
    /// A request is admitted only when the KV pools can hold everything it may
    /// generate alongside the requests already running, so a GPU kernel's pool
    /// never runs dry mid-decode. Otherwise it waits at the head of the queue,
    /// keeping arrival order, until finished requests hand their blocks back.
    fn admit(&mut self) -> Result<Vec<u64>> {
        let mut zero_finished = Vec::new();
        while self.active.len() < self.max_batch {
            let Some(front) = self.pending.front() else {
                break;
            };
            let reserved = self.reservation(front);
            if self.shares_a_prompt_in_flight(&front.prompt) {
                break;
            }
            if front.max_new_tokens > 0 {
                // Cached prefixes hold KV blocks too. Give them back, oldest first,
                // before making a request wait for running ones to finish.
                while !self.kv_fits(reserved)
                    && self
                        .prefix_cache
                        .as_mut()
                        .is_some_and(PrefixCache::evict_oldest)
                {}
                if !self.kv_fits(reserved) {
                    break;
                }
            }
            let p = self.pending.pop_front().expect("front exists");
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
                    p.sampler,
                )?),
                None => {
                    // Resume from the longest cached prefix if one exists;
                    // otherwise prefill from scratch. Either way, cache this
                    // prompt's prefix so later requests can extend it.
                    let align = if self.generator.kv_free_tokens().is_some() {
                        KV_BLOCK_TOKENS
                    } else {
                        1
                    };
                    let resume = self
                        .prefix_cache
                        .as_ref()
                        .and_then(|c| c.longest_prefix(&p.prompt, align));
                    // The prompt goes in over the next ticks (see `step`).
                    let session = match resume {
                        Some(snap) => {
                            self.resume_hits += 1;
                            // Pass the whole prompt so the repetition penalty covers
                            // the cached prefix too (resume derives the suffix).
                            self.generator
                                .resume_session_deferred(snap, &p.prompt, p.sampler)?
                        }
                        None => self
                            .generator
                            .start_session_deferred(&p.prompt, p.sampler)?,
                    };
                    Decoder::Plain(session)
                }
            };
            let cache_prompt = match (&decoder, &self.prefix_cache) {
                (Decoder::Plain(_), Some(_)) => Some(p.prompt),
                _ => None,
            };
            self.active.push(Active {
                id: p.id,
                decoder,
                remaining: p.max_new_tokens,
                eos: p.eos,
                reserved,
                cache_prompt,
            });
        }
        Ok(zero_finished)
    }

    /// Feed admitted prompts into the model, oldest request first. While any
    /// request is decoding, prompts go in [`PREFILL_PIECE`] tokens at a time and
    /// stop once [`PREFILL_TICK_BUDGET`] is spent, so running streams keep
    /// moving. With nothing decoding there is nothing to protect, and prompts go
    /// in whole. A prompt that finishes has its prefix cached.
    fn feed_prompts(&mut self) -> Result<()> {
        let decoding = self.active.iter().any(|a| match &a.decoder {
            Decoder::Plain(s) => !s.is_prefilling(),
            Decoder::Speculative(_) => true,
        });
        let piece = if decoding { PREFILL_PIECE } else { usize::MAX };
        let deadline = Instant::now() + PREFILL_TICK_BUDGET;
        let align = if self.generator.kv_free_tokens().is_some() {
            KV_BLOCK_TOKENS
        } else {
            1
        };
        for a in &mut self.active {
            let Decoder::Plain(session) = &mut a.decoder else {
                continue;
            };
            while session.is_prefilling() {
                if decoding && Instant::now() >= deadline {
                    return Ok(());
                }
                session.prefill_some(piece)?;
            }
            // On a kernel that pages KV, the cached prefix is the prompt cut back
            // to whole blocks, so the snapshot shares those blocks instead of
            // copying the prompt's KV back to the host. Elsewhere it is a host
            // copy of the whole prompt.
            if let (Some(prompt), Some(cache)) = (a.cache_prompt.take(), self.prefix_cache.as_mut())
            {
                let aligned = prompt.len() / align * align;
                if aligned > 0 {
                    cache.insert(
                        prompt[..aligned].to_vec(),
                        session.snapshot_prefix(aligned)?,
                    );
                }
            }
        }
        Ok(())
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
        self.feed_prompts()?;

        // Every plain slot with its prompt in advances in one batched pass.
        let mut plain: Vec<&mut GenerationSession<'a, K>> = self
            .active
            .iter_mut()
            .filter_map(|a| match &mut a.decoder {
                Decoder::Plain(s) if !s.is_prefilling() => Some(s),
                _ => None,
            })
            .collect();
        let mut plain_tokens = self.generator.step_sessions(&mut plain)?.into_iter();

        let mut still_active = Vec::with_capacity(self.active.len());
        for mut a in self.active.drain(..) {
            // Plain slots yield one token; speculative slots yield a whole
            // round's accepted tokens (never more than `remaining`).
            let emitted = match &mut a.decoder {
                Decoder::Plain(s) if s.is_prefilling() => {
                    still_active.push(a);
                    continue;
                }
                Decoder::Plain(_) => vec![plain_tokens.next().expect("one token per plain slot")],
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

    /// A throwaway snapshot of `rows` positions; these tests exercise the cache's
    /// bookkeeping, not the KV contents.
    fn snap_of(rows: usize) -> KvSnapshot {
        let kv_cfg = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            block_size: 16,
        };
        let mut orch = ForwardOrchestrator::new(
            StubKernel::new(1, 4, 2),
            PagedKvCache::new(kv_cfg, 4),
            crate::forward::KvQuant::None,
        );
        for _ in 0..rows {
            orch.decode_token(&mut [0.0; 4]).unwrap();
        }
        orch.snapshot()
    }

    fn snap() -> KvSnapshot {
        snap_of(0)
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
        cache.insert(vec![1], snap_of(1));
        cache.insert(vec![1, 2, 3], snap_of(3));
        cache.insert(vec![1, 2], snap_of(2));
        cache.insert(vec![9, 9], snap_of(2));

        let rows =
            |prompt: &[u32], align| cache.longest_prefix(prompt, align).map(|s| s.position());
        // Longest shared run with [1,2,3,4] is [1,2,3].
        assert_eq!(rows(&[1, 2, 3, 4], 1), Some(3));
        // Shared but diverging: [1,2,7] shares [1,2] with the cached [1,2,3].
        assert_eq!(rows(&[1, 2, 7], 1), Some(2));
        // The whole prompt is never reused: one token must be left to prefill.
        assert_eq!(rows(&[1, 2, 3], 1), Some(2));
        assert_eq!(rows(&[1], 1), None);
        // Nothing shared at all.
        assert_eq!(rows(&[5, 5, 5], 1), None);
        // Cut down to whole blocks: two shared rows make no whole block of 2...
        assert_eq!(rows(&[1, 2, 3, 4], 2), Some(2));
        // ...and none of 4.
        assert_eq!(rows(&[1, 2, 3, 4], 4), None);
    }
}
