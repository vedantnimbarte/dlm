//! Speculative decoding.
//!
//! Something cheap proposes `gamma` tokens; the large **target**
//! model scores all of them in **one** forward pass (a [`prefill`] over the
//! proposals) and keeps the longest prefix it agrees with. A round yields between
//! 1 and `gamma + 1` tokens for a single target pass, so it is faster than plain
//! decoding whenever the draft guesses well and the target pass over `gamma + 1`
//! tokens costs less than `gamma + 1` single-token passes — which the batched
//! prefill kernels make true.
//!
//! Proposals come from a drafter: a small **draft model** sharing the
//! target's vocabulary, or **n-gram lookup** ([`SpeculativeSession::ngram`]), which needs no
//! second model at all — it finds the context's latest suffix that occurred
//! earlier and proposes what followed it. Copying is what lookup catches: code
//! being rewritten, a document being quoted, a list being continued. It pays most
//! on a streamed model, where a pass costs moving every layer's weights, so
//! verifying several tokens costs about what generating one does.
//!
//! Acceptance is the standard rejection rule (Leviathan et al., Chen et al.):
//! draft token `x`, drawn from the draft distribution `q`, is kept with
//! probability `min(1, p(x)/q(x))` under the target distribution `p`; on the
//! first rejection the round ends with a token drawn from `max(0, p − q)`,
//! renormalized; if every proposal is kept, a bonus token is drawn from the
//! target's next distribution. Each emitted token is then distributed exactly as
//! the target's own sampler would have drawn it. A lookup proposal is
//! deterministic, a one-hot `q`, so the same rule keeps it with probability
//! `p(x)` and on rejection draws from `p` with `x` removed. Both distributions are the
//! request's [`Sampler`] applied to each model's logits (temperature, top-k/p,
//! min-p, repetition penalty), and with [`Sampler::Greedy`] both are one-hot, so
//! the rule reduces to "keep while the draft matches the target's argmax" and the
//! output is **identical** to plain target-greedy decoding. A sampled run follows
//! the target's distribution, but not the same draws as a plain seeded run.
//!
//! Both models keep their KV cache across rounds. After a round each is
//! [`truncate`]d back to the tokens now committed, so a rejection costs a
//! rollback, not a re-prefill. A round with nothing to propose (lookup found no
//! match) is a plain single-token step, not a verify pass over one token.
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

/// Where a round's proposals come from.
trait Drafter {
    /// Up to `gamma` tokens continuing `seq`, each with the distribution it was
    /// drawn from. `seen` is `seq` as a set, for the repetition penalty. Empty
    /// when there is nothing worth verifying.
    fn propose(
        &mut self,
        seq: &[u32],
        seen: &HashSet<u32>,
        gamma: usize,
        sampler: &Sampler,
        rng: &mut SplitMix64,
    ) -> Result<Vec<(u32, Vec<(u32, f32)>)>>;

    /// Roll back to the first `len` tokens of the sequence.
    fn truncate(&mut self, _len: usize) {}

    /// KV rows the drafter holds.
    fn kv_rows(&self) -> usize {
        0
    }
}

/// A draft model sampling its proposals, with its own KV cache.
struct ModelDrafter<'a, D: ComputeKernel> {
    model: &'a Generator<D>,
    kv: ForwardOrchestrator<&'a D>,
}

impl<D: ComputeKernel> Drafter for ModelDrafter<'_, D> {
    fn propose(
        &mut self,
        seq: &[u32],
        seen: &HashSet<u32>,
        gamma: usize,
        sampler: &Sampler,
        rng: &mut SplitMix64,
    ) -> Result<Vec<(u32, Vec<(u32, f32)>)>> {
        // Catch up on the tokens it has not seen (at least the pending one),
        // then propose `gamma` tokens from its own distribution.
        let unseen = self.kv.position();
        let mut hidden = self.model.prefill(&mut self.kv, &seq[unseen..])?;
        let mut seen = seen.clone();
        let mut proposals: Vec<(u32, Vec<(u32, f32)>)> = Vec::with_capacity(gamma);
        for j in 0..gamma {
            if j > 0 {
                hidden = self.model.embed(proposals[j - 1].0, self.kv.position())?;
                self.kv.decode_token(&mut hidden)?;
            }
            let q = self.model.distribution(&hidden, &seen, sampler)?;
            let x = sample_from(&q, rng);
            seen.insert(x);
            proposals.push((x, q));
        }
        Ok(proposals)
    }

    fn truncate(&mut self, len: usize) {
        self.kv.truncate(len);
    }

    fn kv_rows(&self) -> usize {
        self.kv.position()
    }
}

/// Longest context suffix n-gram lookup tries. Longer matches predict better;
/// past a few tokens they mostly find the same place a shorter one does.
const NGRAM_MAX: usize = 3;

/// Prompt lookup: proposes what followed the latest earlier occurrence of the
/// context's longest recurring suffix. Deterministic, so each proposal's
/// distribution is one-hot.
struct NgramDrafter;

impl Drafter for NgramDrafter {
    fn propose(
        &mut self,
        seq: &[u32],
        _seen: &HashSet<u32>,
        gamma: usize,
        _sampler: &Sampler,
        _rng: &mut SplitMix64,
    ) -> Result<Vec<(u32, Vec<(u32, f32)>)>> {
        Ok(ngram_lookup(seq, gamma)
            .iter()
            .map(|&x| (x, vec![(x, 1.0)]))
            .collect())
    }
}

/// Up to `gamma` tokens that followed the most recent earlier occurrence of
/// `seq`'s longest suffix of at most [`NGRAM_MAX`] tokens; empty if none recurs.
// ponytail: a linear scan per round, O(context). A round is a full model pass,
// so this is noise at any context dlm serves; index n-grams in a map if it ever
// shows in a profile.
fn ngram_lookup(seq: &[u32], gamma: usize) -> &[u32] {
    for n in (1..=NGRAM_MAX.min(seq.len().saturating_sub(1))).rev() {
        let suffix = &seq[seq.len() - n..];
        if let Some(at) = (0..seq.len() - n).rev().find(|&i| &seq[i..i + n] == suffix) {
            let from = at + n;
            return &seq[from..(from + gamma).min(seq.len())];
        }
    }
    &[]
}

/// A resumable speculative decoder: one [`step`](Self::step) runs a single
/// propose / target-verify round and returns the tokens it produced.
///
/// Invariant between rounds: `seq` holds the prompt and every emitted token, and
/// the target's KV cache (and a draft model's, once caught up) holds all of `seq`
/// **except its last token**, which is fed at the start of the next round.
/// Borrows the target, so many sessions can be driven concurrently by a
/// scheduler.
pub struct SpeculativeSession<'a, T: ComputeKernel> {
    target: &'a Generator<T>,
    target_kv: ForwardOrchestrator<&'a T>,
    drafter: Box<dyn Drafter + 'a>,
    gamma: usize,
    sampler: Sampler,
    rng: SplitMix64,
    seq: Vec<u32>,
    /// Tokens in `seq`, for the repetition penalty.
    seen: HashSet<u32>,
    proposed: usize,
    accepted: usize,
}

impl<'a, T: ComputeKernel> SpeculativeSession<'a, T> {
    /// Begin a session where `draft` proposes `gamma` tokens per round,
    /// prefilling `prompt` (all but its last token) into both models.
    pub fn new<D: ComputeKernel>(
        target: &'a Generator<T>,
        draft: &'a Generator<D>,
        gamma: usize,
        prompt: &[u32],
        sampler: Sampler,
    ) -> Result<Self> {
        let mut kv = draft.orchestrator();
        if let Some((_, head)) = prompt.split_last().filter(|(_, h)| !h.is_empty()) {
            draft.prefill(&mut kv, head)?;
        }
        let drafter = ModelDrafter { model: draft, kv };
        Self::with_drafter(target, Box::new(drafter), gamma, prompt, sampler)
    }

    /// Begin a session that proposes up to `gamma` tokens per round by n-gram
    /// lookup in its own context, with no draft model.
    pub fn ngram(
        target: &'a Generator<T>,
        gamma: usize,
        prompt: &[u32],
        sampler: Sampler,
    ) -> Result<Self> {
        Self::with_drafter(target, Box::new(NgramDrafter), gamma, prompt, sampler)
    }

    fn with_drafter(
        target: &'a Generator<T>,
        drafter: Box<dyn Drafter + 'a>,
        gamma: usize,
        prompt: &[u32],
        sampler: Sampler,
    ) -> Result<Self> {
        let Some((_, head)) = prompt.split_last() else {
            return Err(DlmError::InvalidConfig("prompt must be non-empty".into()));
        };
        let mut target_kv = target.orchestrator();
        if !head.is_empty() {
            target.prefill(&mut target_kv, head)?;
        }
        Ok(Self {
            target,
            target_kv,
            drafter,
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

        // 1. The drafter proposes. Nothing to propose makes this a plain step.
        let (proposals, draft_dists): (Vec<u32>, Vec<_>) = self
            .drafter
            .propose(&self.seq, &self.seen, gamma, &self.sampler, &mut self.rng)?
            .into_iter()
            .unzip();
        if proposals.is_empty() {
            return Self::step_together(&mut [self]);
        }
        let gamma = proposals.len();
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
            let (token, kept) = judge(&p, q, x, &mut self.rng);
            emitted.push(token);
            if !kept {
                break;
            }
            seen.insert(x);
            self.accepted += 1;
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
        self.drafter.truncate(keep);
        Ok(emitted)
    }

    /// Advance each session by one token without speculating, through one
    /// [`decode_batch`](ForwardOrchestrator::decode_batch) pass: what a round with
    /// no proposals comes to, and how a scheduler keeps several sessions in the
    /// one pass a batching kernel fuses. Every session must share one target.
    pub(crate) fn step_together(sessions: &mut [&mut Self]) -> Result<Vec<u32>> {
        let mut hiddens = sessions
            .iter()
            .map(|s| {
                let pending = *s.seq.last().expect("seq holds the prompt");
                s.target.embed(pending, s.target_kv.position())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut orchs: Vec<_> = sessions.iter_mut().map(|s| &mut s.target_kv).collect();
        let mut views: Vec<&mut [f32]> = hiddens.iter_mut().map(Vec::as_mut_slice).collect();
        ForwardOrchestrator::decode_batch(&mut orchs, &mut views)?;
        sessions
            .iter_mut()
            .zip(&hiddens)
            .map(|(s, hidden)| {
                let p = s.target.distribution(hidden, &s.seen, &s.sampler)?;
                let next = sample_from(&p, &mut s.rng);
                s.seq.push(next);
                s.seen.insert(next);
                Ok(next)
            })
            .collect()
    }

    /// KV rows the session holds in its larger cache (a draft model runs up to
    /// `gamma` tokens ahead of the target within a round).
    pub fn kv_rows(&self) -> usize {
        self.target_kv.position().max(self.drafter.kv_rows())
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

/// The rejection rule for one proposal: keep `x`, drawn from `q`, with
/// probability `min(1, p(x)/q(x))`; otherwise replace it with a draw from the
/// residual. Returns the emitted token and whether it was `x` kept. Either way
/// the token is distributed as `p`.
fn judge(p: &[(u32, f32)], q: &[(u32, f32)], x: u32, rng: &mut SplitMix64) -> (u32, bool) {
    let (px, qx) = (prob_of(p, x), prob_of(q, x));
    if qx > 0.0 && rng.next_f32() < px / qx {
        (x, true)
    } else {
        (sample_from(&residual(p, q), rng), false)
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

    /// A lookup proposal is a one-hot `q`: kept with probability `p(x)`, and on
    /// rejection replaced from `p` with `x` removed — so what comes out is
    /// distributed exactly as `p`, whichever token was proposed.
    #[test]
    fn a_fixed_proposal_still_emits_the_target_distribution() {
        let p = [(0u32, 0.5f32), (1, 0.3), (2, 0.2)];
        let runs = 200_000;
        let mut rng = SplitMix64::new(7);
        for x in [0u32, 2, 9] {
            let (mut counts, mut kept) = ([0usize; 3], 0usize);
            for _ in 0..runs {
                let (token, k) = judge(&p, &[(x, 1.0)], x, &mut rng);
                counts[token as usize] += 1;
                kept += k as usize;
            }
            for &(t, pt) in &p {
                let got = counts[t as usize] as f32 / runs as f32;
                assert!(
                    (got - pt).abs() < 0.01,
                    "proposal {x}: token {t} at {got}, want {pt}"
                );
            }
            let want = prob_of(&p, x);
            let got = kept as f32 / runs as f32;
            assert!(
                (got - want).abs() < 0.01,
                "proposal {x}: kept {got}, want {want}"
            );
        }
    }

    /// The longest recurring suffix wins, its latest occurrence is used, and the
    /// proposal stops at `gamma` or the end of the context.
    #[test]
    fn lookup_continues_the_latest_longest_match() {
        // Suffix [1, 2, 3] occurred at 0; [2, 3] alone more recently (at 5).
        let seq = [1, 2, 3, 4, 9, 2, 3, 7, 1, 2, 3];
        assert_eq!(ngram_lookup(&seq, 2), &[4, 9]);
        // Only a 1-gram recurs: [5] last seen at 2, followed by [6, 5].
        assert_eq!(ngram_lookup(&[8, 0, 5, 6, 5], 4), &[6, 5]);
        // The continuation runs out at the end of the context.
        assert_eq!(ngram_lookup(&[4, 4, 4], 8), &[4]);
        // Nothing recurs, or nothing before the suffix: no proposal.
        assert!(ngram_lookup(&[1, 2, 3], 4).is_empty());
        assert!(ngram_lookup(&[1], 4).is_empty());
    }
}
