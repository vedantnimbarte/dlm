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
use crate::forward::ComputeKernel;
use crate::generate::{GenerationSession, Generator, Sampler};
use crate::speculative::SpeculativeSession;
use std::collections::VecDeque;

/// A queued request awaiting admission.
struct Pending {
    id: u64,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    eos: Option<u32>,
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
    eos: Option<u32>,
}

/// A completed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finished {
    pub id: u64,
    pub tokens: Vec<u32>,
}

/// What one scheduler tick produced: `(request id, token)` pairs emitted this
/// step, and the ids of requests that finished (their last token is in
/// `produced`). Streaming consumers forward `produced` and close on `finished`.
#[derive(Debug, Clone, Default)]
pub struct Tick {
    pub produced: Vec<(u64, u32)>,
    pub finished: Vec<u64>,
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
        }
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
        }
    }

    /// Cumulative `(proposed, accepted)` draft-token counts over all *retired*
    /// speculative slots. Zero unless the scheduler was built with a draft.
    pub fn speculative_stats(&self) -> (usize, usize) {
        (self.proposed, self.accepted)
    }

    /// Queue a request. Errors on an empty prompt.
    pub fn submit(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        eos: Option<u32>,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(crate::error::FlipError::InvalidConfig("prompt is empty".into()));
        }
        self.pending.push_back(Pending {
            id,
            prompt,
            max_new_tokens,
            eos,
        });
        Ok(())
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
                None => Decoder::Plain(self.generator.start_session(&p.prompt, Sampler::Greedy)?),
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
        let mut tick = Tick::default();
        tick.finished = zero_finished;
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
                if a.remaining == 0 || Some(token) == a.eos {
                    done = true;
                    break;
                }
            }

            if done {
                if let Decoder::Speculative(s) = &a.decoder {
                    self.proposed += s.proposed();
                    self.accepted += s.accepted();
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
