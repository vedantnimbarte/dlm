//! Speculative decoding.
//!
//! A small, cheap **draft** model proposes `gamma` tokens; the large **target**
//! model scores all of them in **one** forward pass (a [`prefill`] over the
//! proposals) and keeps the longest prefix it agrees with. A round yields between
//! 1 and `gamma + 1` tokens for a single target pass, so it is faster than plain
//! decoding whenever the draft guesses well and the target pass over `gamma + 1`
//! tokens costs less than `gamma + 1` single-token passes — which the batched
//! prefill kernels make true.
//!
//! Acceptance is the standard rejection rule (Leviathan et al., Chen et al.):
//! draft token `x`, drawn from the draft distribution `q`, is kept with
//! probability `min(1, p(x)/q(x))` under the target distribution `p`; on the
//! first rejection the round ends with a token drawn from `max(0, p − q)`,
//! renormalized; if every proposal is kept, a bonus token is drawn from the
//! target's next distribution. Each emitted token is then distributed exactly as
//! the target's own sampler would have drawn it. Both distributions are the
//! request's [`Sampler`] applied to each model's logits (temperature, top-k/p,
//! min-p, repetition penalty), and with [`Sampler::Greedy`] both are one-hot, so
//! the rule reduces to "keep while the draft matches the target's argmax" and the
//! output is **identical** to plain target-greedy decoding. A sampled run follows
//! the target's distribution, but not the same draws as a plain seeded run.
//!
//! Both models keep their KV cache across rounds. After a round each is
//! [`truncate`]d back to the tokens now committed, so a rejection costs a
//! rollback, not a re-prefill.
//!
//! Two entry points share the same round logic:
//!
//! * [`SpeculativeDecoder`] — a one-shot greedy `prompt → tokens` decoder.
//! * [`SpeculativeSession`] — a *resumable* decoder that runs a single round per
//!   [`step`](SpeculativeSession::step) call, emitting the 1..=`gamma`+1 tokens
//!   that round produced. This is the surface the continuous-batching engine
//!   drives ([`BatchScheduler::with_speculative`](crate::batching::BatchScheduler::with_speculative)).
//!
//! [`prefill`]: crate::forward::ForwardOrchestrator::prefill
//! [`truncate`]: crate::forward::ForwardOrchestrator::truncate

use crate::error::{DlmError, Result};
use crate::forward::{ComputeKernel, ForwardOrchestrator};
use crate::generate::{sample_from, Generator, Sampler, SplitMix64};
use std::collections::HashSet;

/// Outcome of a speculative generation.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// The generated continuation (exactly target-greedy).
    pub tokens: Vec<u32>,
    /// Draft tokens proposed across all rounds.
    pub proposed: usize,
    /// Draft tokens accepted by the target.
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

    /// Greedily generate up to `max_new_tokens` tokens for `prompt`.
    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Result<SpeculativeResult> {
        let mut session = SpeculativeSession::new(
            &self.target,
            &self.draft,
            self.gamma,
            prompt,
            Sampler::Greedy,
        )?;
        let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);
        while out.len() < max_new_tokens {
            // Cap each round to the tokens still wanted so the length limit can't
            // masquerade as a draft rejection.
            let emitted = session.step(max_new_tokens - out.len())?;
            if emitted.is_empty() {
                break; // unreachable for a non-zero budget; a defensive progress guard
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
/// Invariant between rounds: `seq` holds the prompt and every emitted token, and
/// both models' KV caches hold all of `seq` **except its last token**, which is
/// fed to them at the start of the next round. Borrows both generators, so many
/// sessions can be driven concurrently by a scheduler.
pub struct SpeculativeSession<'a, T: ComputeKernel, D: ComputeKernel> {
    target: &'a Generator<T>,
    draft: &'a Generator<D>,
    target_kv: ForwardOrchestrator<&'a T>,
    draft_kv: ForwardOrchestrator<&'a D>,
    gamma: usize,
    sampler: Sampler,
    rng: SplitMix64,
    seq: Vec<u32>,
    /// Tokens in `seq`, for the repetition penalty.
    seen: HashSet<u32>,
    proposed: usize,
    accepted: usize,
}

impl<'a, T: ComputeKernel, D: ComputeKernel> SpeculativeSession<'a, T, D> {
    /// Begin a session that proposes `gamma` tokens per round, prefilling
    /// `prompt` (all but its last token) into both models.
    pub fn new(
        target: &'a Generator<T>,
        draft: &'a Generator<D>,
        gamma: usize,
        prompt: &[u32],
        sampler: Sampler,
    ) -> Result<Self> {
        let Some((_, head)) = prompt.split_last() else {
            return Err(DlmError::InvalidConfig("prompt must be non-empty".into()));
        };
        let mut target_kv = target.orchestrator();
        let mut draft_kv = draft.orchestrator();
        if !head.is_empty() {
            target.prefill(&mut target_kv, head)?;
            draft.prefill(&mut draft_kv, head)?;
        }
        Ok(Self {
            target,
            draft,
            target_kv,
            draft_kv,
            gamma: gamma.max(1),
            rng: SplitMix64::new(sampler.seed()),
            sampler,
            seq: prompt.to_vec(),
            seen: prompt.iter().copied().collect(),
            proposed: 0,
            accepted: 0,
        })
    }

    /// Run one speculative round, emitting at most `budget` tokens (normally
    /// 1..=`gamma`+1) and appending them to the sequence. The caller decides when
    /// to stop (length or EOS). For `budget >= 1` the result is never empty.
    pub fn step(&mut self, budget: usize) -> Result<Vec<u32>> {
        if budget == 0 {
            return Ok(Vec::new());
        }
        let gamma = self.gamma.min(budget);

        // 1. The draft catches up on the tokens it has not seen (at least the
        //    pending one), then proposes `gamma` tokens from its own distribution.
        let unseen = self.draft_kv.position();
        let mut hidden = self
            .draft
            .prefill(&mut self.draft_kv, &self.seq[unseen..])?;
        let mut seen = self.seen.clone();
        let mut proposals = Vec::with_capacity(gamma);
        let mut draft_dists = Vec::with_capacity(gamma);
        for j in 0..gamma {
            if j > 0 {
                hidden = self
                    .draft
                    .embed(proposals[j - 1], self.draft_kv.position())?;
                self.draft_kv.decode_token(&mut hidden)?;
            }
            let q = self.draft.distribution(&hidden, &seen, &self.sampler)?;
            let x = sample_from(&q, &mut self.rng);
            seen.insert(x);
            proposals.push(x);
            draft_dists.push(q);
        }
        self.proposed += gamma;

        // 2. The target scores the pending token and every proposal in one pass.
        //    The last `gamma + 1` hidden states give its distribution before each
        //    proposal and after the last.
        let start = self.target_kv.position();
        let fed: Vec<u32> = self.seq[start..]
            .iter()
            .chain(&proposals)
            .copied()
            .collect();
        let mut hiddens = Vec::new();
        for (i, &token) in fed.iter().enumerate() {
            hiddens.extend(self.target.embed(token, start + i)?);
        }
        let h = hiddens.len() / fed.len();
        self.target_kv.prefill(&mut hiddens)?;
        let target_hidden = |j: usize| {
            let at = (fed.len() - gamma - 1 + j) * h;
            &hiddens[at..at + h]
        };

        // 3. Accept each proposal with probability min(1, p/q); on the first
        //    rejection, draw the replacement from the residual and stop.
        let mut seen = self.seen.clone();
        let mut emitted = Vec::with_capacity(gamma + 1);
        let accepted_before = self.accepted;
        for (j, (&x, q)) in proposals.iter().zip(&draft_dists).enumerate() {
            let p = self
                .target
                .distribution(target_hidden(j), &seen, &self.sampler)?;
            let (px, qx) = (prob_of(&p, x), prob_of(q, x));
            if qx > 0.0 && self.rng.next_f32() < px / qx {
                emitted.push(x);
                seen.insert(x);
                self.accepted += 1;
            } else {
                emitted.push(sample_from(&residual(&p, q), &mut self.rng));
                break;
            }
        }
        // Only when every proposal was accepted: a rejection at the last one
        // also leaves `gamma` tokens emitted, and must not draw a bonus.
        let all_accepted = self.accepted - accepted_before == gamma;
        if all_accepted && emitted.len() < budget {
            let p = self
                .target
                .distribution(target_hidden(gamma), &seen, &self.sampler)?;
            emitted.push(sample_from(&p, &mut self.rng));
        }

        // 4. Commit, and roll both caches back to everything but the new pending
        //    token. What stays is valid: every emitted token but the last is an
        //    accepted proposal, i.e. exactly what both models were fed.
        self.seq.extend(&emitted);
        self.seen.extend(&emitted);
        let keep = self.seq.len() - 1;
        self.target_kv.truncate(keep);
        self.draft_kv.truncate(keep);
        Ok(emitted)
    }

    /// Draft tokens proposed so far.
    /// KV rows the session holds in its larger cache (the draft runs up to
    /// `gamma` tokens ahead of the target within a round).
    pub fn kv_rows(&self) -> usize {
        self.target_kv.position().max(self.draft_kv.position())
    }

    pub fn proposed(&self) -> usize {
        self.proposed
    }

    /// Draft tokens accepted so far.
    pub fn accepted(&self) -> usize {
        self.accepted
    }
}

/// The probability `dist` assigns to `token` (0 outside its support).
fn prob_of(dist: &[(u32, f32)], token: u32) -> f32 {
    dist.iter()
        .find(|&&(t, _)| t == token)
        .map_or(0.0, |&(_, p)| p)
}

/// `max(0, p − q)`, renormalized: the distribution a rejected position resamples
/// from. When rounding leaves no mass (p and q agree everywhere), `p` itself.
fn residual(p: &[(u32, f32)], q: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let qmap: std::collections::HashMap<u32, f32> = q.iter().copied().collect();
    let mut r: Vec<(u32, f32)> = p
        .iter()
        .map(|&(t, pt)| (t, (pt - qmap.get(&t).copied().unwrap_or(0.0)).max(0.0)))
        .filter(|&(_, m)| m > 0.0)
        .collect();
    let total: f32 = r.iter().map(|&(_, m)| m).sum();
    if total <= 0.0 {
        return p.to_vec();
    }
    for (_, m) in &mut r {
        *m /= total;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-hot distributions make the residual exactly the target's argmax, which
    /// is what keeps greedy speculation identical to greedy decoding.
    #[test]
    fn residual_of_one_hots_is_the_target_token() {
        assert_eq!(residual(&[(7, 1.0)], &[(3, 1.0)]), vec![(7, 1.0)]);
        // Agreeing distributions leave no residual mass; fall back to p.
        assert_eq!(residual(&[(7, 1.0)], &[(7, 1.0)]), vec![(7, 1.0)]);
        let r = residual(&[(1, 0.5), (2, 0.5)], &[(1, 0.8), (2, 0.2)]);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 2);
        assert!((r[0].1 - 1.0).abs() < 1e-6);
    }
}
