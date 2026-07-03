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

use crate::error::Result;
use crate::forward::ComputeKernel;
use crate::generate::{GenerationSession, Generator, Sampler};
use std::collections::VecDeque;

/// A queued request awaiting admission.
struct Pending {
    id: u64,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    eos: Option<u32>,
}

/// An in-flight generation occupying a batch slot.
struct Active<'a, K: ComputeKernel> {
    id: u64,
    session: GenerationSession<'a, K>,
    remaining: usize,
    eos: Option<u32>,
    output: Vec<u32>,
}

/// A completed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finished {
    pub id: u64,
    pub tokens: Vec<u32>,
}

/// A continuous-batching scheduler over a borrowed generator.
pub struct BatchScheduler<'a, K: ComputeKernel> {
    generator: &'a Generator<K>,
    max_batch: usize,
    pending: VecDeque<Pending>,
    active: Vec<Active<'a, K>>,
    finished: Vec<Finished>,
}

impl<'a, K: ComputeKernel> BatchScheduler<'a, K> {
    /// Create a scheduler running at most `max_batch` concurrent generations.
    pub fn new(generator: &'a Generator<K>, max_batch: usize) -> Self {
        Self {
            generator,
            max_batch: max_batch.max(1),
            pending: VecDeque::new(),
            active: Vec::new(),
            finished: Vec::new(),
        }
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
    fn admit(&mut self) -> Result<()> {
        while self.active.len() < self.max_batch {
            let Some(p) = self.pending.pop_front() else { break };
            if p.max_new_tokens == 0 {
                self.finished.push(Finished {
                    id: p.id,
                    tokens: Vec::new(),
                });
                continue;
            }
            let session = self.generator.start_session(&p.prompt, Sampler::Greedy)?;
            self.active.push(Active {
                id: p.id,
                session,
                remaining: p.max_new_tokens,
                eos: p.eos,
                output: Vec::new(),
            });
        }
        Ok(())
    }

    /// One scheduler tick: admit, advance every active session by one token,
    /// retire any that finished.
    pub fn step(&mut self) -> Result<()> {
        self.admit()?;
        let mut still_active = Vec::with_capacity(self.active.len());
        for mut a in self.active.drain(..) {
            let token = a.session.step()?;
            a.output.push(token);
            a.remaining -= 1;
            let done = a.remaining == 0 || Some(token) == a.eos;
            if done {
                self.finished.push(Finished {
                    id: a.id,
                    tokens: std::mem::take(&mut a.output),
                });
            } else {
                still_active.push(a);
            }
        }
        self.active = still_active;
        Ok(())
    }

    /// Run ticks until every request has completed, returning the results (in
    /// completion order).
    pub fn run(&mut self) -> Result<Vec<Finished>> {
        while self.has_work() {
            self.step()?;
        }
        Ok(std::mem::take(&mut self.finished))
    }
}
