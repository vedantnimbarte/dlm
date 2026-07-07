//! Speculative decoding (`PRD.md` §3.3).
//!
//! A small, cheap **draft** model proposes `gamma` tokens; the large **target**
//! model verifies them. With greedy sampling the rule is exact: accept each draft
//! token while it equals the target's greedy choice; at the first mismatch take
//! the target's token instead; if all `gamma` are accepted, append the target's
//! next ("bonus") token. This yields between 1 and `gamma + 1` tokens per round
//! and — crucially — produces **exactly** the same sequence as plain target-greedy
//! decoding, just faster when the draft guesses well.
//!
//! The speed win comes from verifying all `gamma` positions in a single *batched*
//! target forward pass. `dlm`'s CPU forward is single-token, so this
//! implementation verifies sequentially (one target step per position) — the
//! accept/reject logic and its exactness are identical; only the wall-clock
//! saving awaits a batched kernel. Acceptance statistics are reported so the
//! benefit is measurable.
//!
//! Two entry points share the same round logic:
//!
//! * [`SpeculativeDecoder`] — a one-shot `prompt → tokens` decoder.
//! * [`SpeculativeSession`] — a *resumable* decoder that runs a single round per
//!   [`step`](SpeculativeSession::step) call, emitting the 1..=`gamma`+1 tokens
//!   that round produced. This is the surface the continuous-batching engine
//!   drives ([`BatchScheduler::with_speculative`](crate::batching::BatchScheduler::with_speculative)):
//!   a scheduler tick advances every request by one round, and the accepted
//!   tokens for each request flow out through the same streaming path as plain
//!   decoding.

use crate::error::Result;
use crate::forward::ComputeKernel;
use crate::generate::{GenerationConfig, Generator, Sampler};

/// Outcome of a speculative generation.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// The generated continuation (exactly target-greedy).
    pub tokens: Vec<u32>,
    /// Draft tokens proposed across all rounds.
    pub proposed: usize,
    /// Draft tokens accepted (matched the target).
    pub accepted: usize,
}

impl SpeculativeResult {
    /// Fraction of proposed draft tokens the target accepted (0.0–1.0).
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }
}

/// A speculative decoder pairing a target and a draft generator.
pub struct SpeculativeDecoder<T: ComputeKernel, D: ComputeKernel> {
    target: Generator<T>,
    draft: Generator<D>,
    gamma: usize,
}

impl<T: ComputeKernel, D: ComputeKernel> SpeculativeDecoder<T, D> {
    /// Pair a `target` with a `draft`, proposing `gamma` tokens per round.
    pub fn new(target: Generator<T>, draft: Generator<D>, gamma: usize) -> Self {
        Self {
            target,
            draft,
            gamma: gamma.max(1),
        }
    }

    /// Generate up to `max_new_tokens` tokens for `prompt`.
    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Result<SpeculativeResult> {
        let mut session = SpeculativeSession::new(&self.target, &self.draft, self.gamma, prompt);
        let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);
        while out.len() < max_new_tokens {
            // Cap each round to the tokens still wanted so the length limit can't
            // masquerade as a draft rejection.
            let emitted = session.step(max_new_tokens - out.len())?;
            if emitted.is_empty() {
                break; // unreachable for gamma >= 1; a defensive progress guard
            }
            out.extend(emitted);
        }
        out.truncate(max_new_tokens);
        Ok(SpeculativeResult {
            tokens: out,
            proposed: session.proposed(),
            accepted: session.accepted(),
        })
    }
}

/// A resumable speculative decoder: one [`step`](Self::step) runs a single
/// draft-propose / target-verify round and returns the tokens it produced.
///
/// State is just the running sequence plus acceptance counters; each round
/// re-derives the target's greedy choices from that sequence (the same
/// sequential verification [`SpeculativeDecoder`] uses), so no KV rewind is
/// needed on a rejection. Borrows both generators, so many sessions can be
/// driven concurrently by a scheduler.
pub struct SpeculativeSession<'a, T: ComputeKernel, D: ComputeKernel> {
    target: &'a Generator<T>,
    draft: &'a Generator<D>,
    gamma: usize,
    seq: Vec<u32>,
    proposed: usize,
    accepted: usize,
}

impl<'a, T: ComputeKernel, D: ComputeKernel> SpeculativeSession<'a, T, D> {
    /// Begin a session that proposes `gamma` tokens per round, seeded with
    /// `prompt`. Emitted tokens are appended to the internal sequence.
    pub fn new(
        target: &'a Generator<T>,
        draft: &'a Generator<D>,
        gamma: usize,
        prompt: &[u32],
    ) -> Self {
        Self {
            target,
            draft,
            gamma: gamma.max(1),
            seq: prompt.to_vec(),
            proposed: 0,
            accepted: 0,
        }
    }

    /// The target's greedy next token given the current sequence.
    fn target_next(&self) -> Result<u32> {
        let cfg = GenerationConfig {
            max_new_tokens: 1,
            eos_token: None,
            sampler: Sampler::Greedy,
        };
        Ok(self.target.generate(&self.seq, &cfg)?[0])
    }

    /// Run one speculative round, emitting at most `budget` tokens (normally
    /// 1..=`gamma`+1). The tokens are exactly the target-greedy continuation of
    /// the current sequence and are appended to it. Returns them for the caller
    /// to forward; the caller decides when to stop (length or EOS). For
    /// `budget >= 1` the result is always non-empty, so a driver makes progress.
    pub fn step(&mut self, budget: usize) -> Result<Vec<u32>> {
        if budget == 0 {
            return Ok(Vec::new());
        }
        // Never propose more than the caller still wants this round.
        let gamma = self.gamma.min(budget);
        let draft_cfg = GenerationConfig {
            max_new_tokens: gamma,
            eos_token: None,
            sampler: Sampler::Greedy,
        };
        let draft_tokens = self.draft.generate(&self.seq, &draft_cfg)?;
        self.proposed += draft_tokens.len();

        let mut emitted = Vec::with_capacity(gamma + 1);
        let mut all_accepted = true;
        for &dt in &draft_tokens {
            let target_tok = self.target_next()?;
            if target_tok == dt {
                self.seq.push(dt);
                emitted.push(dt);
                self.accepted += 1;
            } else {
                // Mismatch: take the target's correction and end the round.
                self.seq.push(target_tok);
                emitted.push(target_tok);
                all_accepted = false;
                break;
            }
        }

        // All draft tokens accepted → append the target's bonus token, unless
        // that would exceed the caller's budget for this round.
        if all_accepted && emitted.len() < budget {
            let bonus = self.target_next()?;
            self.seq.push(bonus);
            emitted.push(bonus);
        }

        Ok(emitted)
    }

    /// Draft tokens proposed so far.
    pub fn proposed(&self) -> usize {
        self.proposed
    }

    /// Draft tokens accepted so far.
    pub fn accepted(&self) -> usize {
        self.accepted
    }
}
