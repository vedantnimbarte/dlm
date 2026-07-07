//! End-to-end CPU token generation.
//!
//! Wraps the streaming forward pass into a full autoregressive loop:
//!
//! ```text
//!   token ─► embedding lookup ─► [ transformer stack via ForwardOrchestrator ]
//!         ─► final RMSNorm ─► LM head ─► logits ─► sample ─► next token ─► …
//! ```
//!
//! The transformer stack is any [`ComputeKernel`] — with the
//! [`CpuKernel`](crate::forward::CpuKernel) this is a complete, if slow,
//! CPU inference path: prompt tokens are prefilled to build the KV history, then
//! new tokens are generated one at a time until `max_new_tokens` or an EOS.
//! Swapping in a GPU kernel makes it real inference with no change here.

use crate::cache::{KvCacheConfig, PagedKvCache};
use crate::error::{DlmError, Result};
use crate::forward::cpu::{matvec, rmsnorm};
use crate::forward::{ComputeKernel, ForwardOrchestrator};

/// Index of the largest logit (greedy pick; first max wins on ties).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

/// A tiny SplitMix64 PRNG for stochastic sampling (no external deps). Seeded per
/// session, so a fixed `seed` makes temperature sampling reproducible.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Token-selection strategy over a logit vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampler {
    /// Deterministic argmax.
    Greedy,
    /// Temperature scaling with optional top-k, nucleus (top-p), and min-p
    /// truncation. `top_k == 0` keeps all tokens; `top_p >= 1.0` disables nucleus
    /// filtering; `min_p <= 0` disables min-p; `temperature <= 0` collapses to
    /// [`Greedy`](Sampler::Greedy).
    TopPK {
        temperature: f32,
        top_p: f32,
        top_k: u32,
        /// Keep only tokens whose probability is at least `min_p` times the most
        /// likely token's probability. Applied after top-k/top-p.
        min_p: f32,
        /// Divide the logit of any already-seen token by this factor (multiply
        /// when negative), discouraging repetition. `1.0` disables it.
        repetition_penalty: f32,
        seed: u64,
    },
}

impl Sampler {
    /// The RNG seed a session should use (0 for the deterministic [`Greedy`]).
    ///
    /// [`Greedy`]: Sampler::Greedy
    pub fn seed(&self) -> u64 {
        match self {
            Sampler::Greedy => 0,
            Sampler::TopPK { seed, .. } => *seed,
        }
    }

    /// The repetition-penalty factor (`1.0` — no penalty — for [`Greedy`]).
    ///
    /// [`Greedy`]: Sampler::Greedy
    pub fn repetition_penalty(&self) -> f32 {
        match self {
            Sampler::Greedy => 1.0,
            Sampler::TopPK { repetition_penalty, .. } => *repetition_penalty,
        }
    }

    /// Pick the next token id from `logits`, advancing `rng` for stochastic
    /// samplers (unused by [`Greedy`](Sampler::Greedy)).
    pub fn sample(&self, logits: &[f32], rng: &mut SplitMix64) -> u32 {
        match *self {
            Sampler::Greedy => argmax(logits),
            Sampler::TopPK { temperature, top_p, top_k, min_p, .. } => {
                if temperature <= 0.0 {
                    return argmax(logits);
                }
                sample_topk_topp(logits, temperature, top_p, top_k as usize, min_p, rng)
            }
        }
    }
}

/// Discourage repetition by penalizing the logits of tokens already in the
/// context (HF convention: divide positive logits by `penalty`, multiply
/// negative ones). A `penalty` of `1.0` is a no-op. Applied in place before
/// sampling.
fn apply_repetition_penalty(logits: &mut [f32], seen: &std::collections::HashSet<u32>, penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &t in seen {
        let i = t as usize;
        if i < logits.len() {
            logits[i] = if logits[i] > 0.0 { logits[i] / penalty } else { logits[i] * penalty };
        }
    }
}

/// Temperature + top-k + nucleus (top-p) sampling. Returns a token index drawn
/// from the filtered, renormalized distribution.
fn sample_topk_topp(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    rng: &mut SplitMix64,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    // Candidate indices sorted by logit, descending.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    // top-k cap.
    let k = if top_k > 0 { top_k.min(idx.len()) } else { idx.len() };
    idx.truncate(k);

    // Softmax with temperature over the retained logits (subtract the max for
    // numerical stability; idx[0] is the global max since we sorted).
    let max_logit = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Nucleus (top-p): keep the smallest prefix whose cumulative mass ≥ top_p.
    if top_p > 0.0 && top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut cut = probs.len();
        for (j, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cut = j + 1;
                break;
            }
        }
        probs.truncate(cut);
        idx.truncate(cut);
        let s: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= s;
        }
    }

    // Min-p: drop tokens less likely than `min_p` × the top token's probability
    // (probs[0] is the max since idx was sorted by logit). Then renormalize.
    if min_p > 0.0 {
        let threshold = min_p * probs[0];
        let cut = probs.iter().take_while(|&&p| p >= threshold).count().max(1);
        probs.truncate(cut);
        idx.truncate(cut);
        let s: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= s;
        }
    }

    // Inverse-CDF sample.
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return idx[j] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

/// Generation parameters.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to emit.
    pub max_new_tokens: usize,
    /// Stop early when this token is produced (it is still included in output).
    pub eos_token: Option<u32>,
    /// Sampling strategy.
    pub sampler: Sampler,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 16,
            eos_token: None,
            sampler: Sampler::Greedy,
        }
    }
}

/// A CPU text generator: a transformer kernel plus the embedding, final norm,
/// and LM head that turn token ids into logits and back.
pub struct Generator<K: ComputeKernel> {
    kernel: K,
    /// Token embedding table, row-major `[vocab, hidden]`.
    embedding: Vec<f32>,
    /// Final pre-head RMSNorm weight, `[hidden]`.
    final_norm: Vec<f32>,
    /// LM head, row-major `[vocab, hidden]` (untied from the embedding here).
    lm_head: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
    rms_eps: f32,
    kv_config: KvCacheConfig,
    kv_total_blocks: u32,
}

impl<K: ComputeKernel> Generator<K> {
    /// Assemble a generator, validating that every table matches the shapes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kernel: K,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
        lm_head: Vec<f32>,
        vocab_size: usize,
        rms_eps: f32,
        kv_config: KvCacheConfig,
        kv_total_blocks: u32,
    ) -> Result<Self> {
        let hidden_size = kernel.hidden_size();
        let checks = [
            ("embedding", embedding.len(), vocab_size * hidden_size),
            ("lm_head", lm_head.len(), vocab_size * hidden_size),
            ("final_norm", final_norm.len(), hidden_size),
        ];
        for (name, got, expected) in checks {
            if got != expected {
                return Err(DlmError::InvalidConfig(format!(
                    "{name}: expected {expected} elements, got {got}"
                )));
            }
        }
        Ok(Self {
            kernel,
            embedding,
            final_norm,
            lm_head,
            vocab_size,
            hidden_size,
            rms_eps,
            kv_config,
            kv_total_blocks,
        })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embed a token id into a fresh hidden vector.
    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        let idx = token as usize;
        if idx >= self.vocab_size {
            return Err(DlmError::InvalidConfig(format!(
                "token {token} out of vocab range {}",
                self.vocab_size
            )));
        }
        let start = idx * self.hidden_size;
        Ok(self.embedding[start..start + self.hidden_size].to_vec())
    }

    /// Project a hidden state to vocabulary logits via final norm + LM head.
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let normed = rmsnorm(hidden, &self.final_norm, self.rms_eps);
        matvec(&self.lm_head, &normed, self.vocab_size, self.hidden_size)
    }

    /// Generate a continuation for `prompt`, returning the newly produced token
    /// ids (the prompt itself is not included).
    ///
    /// Prefills the prompt into a fresh KV history, then decodes greedily/by the
    /// configured sampler until `max_new_tokens` or an EOS token.
    pub fn generate(&self, prompt: &[u32], cfg: &GenerationConfig) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(DlmError::InvalidConfig("prompt must be non-empty".into()));
        }

        // Fresh KV state per generation (single sequence).
        let budget = PagedKvCache::new(self.kv_config, self.kv_total_blocks);
        let mut orch = ForwardOrchestrator::new(&self.kernel, budget);

        // Prefill: run every prompt token, carrying the last hidden state.
        let mut hidden = vec![0.0f32; self.hidden_size];
        for &token in prompt {
            hidden = self.embed(token)?;
            orch.decode_token(&mut hidden)?;
        }

        // Decode loop. `seen` tracks the full context (prompt + generated) for
        // the repetition penalty.
        let mut rng = SplitMix64::new(cfg.sampler.seed());
        let penalty = cfg.sampler.repetition_penalty();
        let mut seen: std::collections::HashSet<u32> = prompt.iter().copied().collect();
        let mut generated = Vec::with_capacity(cfg.max_new_tokens);
        for _ in 0..cfg.max_new_tokens {
            let mut logits = self.logits(&hidden);
            apply_repetition_penalty(&mut logits, &seen, penalty);
            let next = cfg.sampler.sample(&logits, &mut rng);
            generated.push(next);
            seen.insert(next);
            if cfg.eos_token == Some(next) {
                break;
            }
            hidden = self.embed(next)?;
            orch.decode_token(&mut hidden)?;
        }
        Ok(generated)
    }
}

/// A resumable, single-token-at-a-time generation, decoupled from the full
/// `generate` loop so a scheduler can interleave many of them (continuous
/// batching). Borrows the generator's weights; each session owns its own KV
/// state, so stepping sessions in any order yields identical per-session output.
pub struct GenerationSession<'a, K: ComputeKernel> {
    generator: &'a Generator<K>,
    orchestrator: crate::forward::ForwardOrchestrator<&'a K>,
    last_hidden: Vec<f32>,
    sampler: Sampler,
    rng: SplitMix64,
    /// Context tokens (prompt + generated) for the repetition penalty.
    seen: std::collections::HashSet<u32>,
}

impl<K: ComputeKernel> Generator<K> {
    /// Begin a step-wise generation: prefills `prompt` and leaves the session
    /// ready to emit the first continuation token via [`GenerationSession::step`].
    pub fn start_session(&self, prompt: &[u32], sampler: Sampler) -> Result<GenerationSession<'_, K>> {
        if prompt.is_empty() {
            return Err(DlmError::InvalidConfig("prompt must be non-empty".into()));
        }
        let budget = crate::cache::PagedKvCache::new(self.kv_config, self.kv_total_blocks);
        let mut orchestrator = crate::forward::ForwardOrchestrator::new(&self.kernel, budget);
        let mut hidden = vec![0.0f32; self.hidden_size];
        for &token in prompt {
            hidden = self.embed(token)?;
            orchestrator.decode_token(&mut hidden)?;
        }
        Ok(GenerationSession {
            generator: self,
            orchestrator,
            last_hidden: hidden,
            rng: SplitMix64::new(sampler.seed()),
            sampler,
            seen: prompt.iter().copied().collect(),
        })
    }

    /// Resume generation from a prior session's [`KvSnapshot`], prefilling only
    /// the `suffix` tokens that follow the snapshotted prefix. The result is
    /// identical to [`start_session`](Self::start_session) on the full
    /// `prefix + suffix` prompt, but skips re-running the shared prefix — the
    /// basis for cross-request prefix caching. `suffix` must be non-empty (the
    /// session needs a last hidden state to produce the first token).
    pub fn resume_session(
        &self,
        snapshot: crate::forward::KvSnapshot,
        suffix: &[u32],
        sampler: Sampler,
    ) -> Result<GenerationSession<'_, K>> {
        if suffix.is_empty() {
            return Err(DlmError::InvalidConfig("resume suffix must be non-empty".into()));
        }
        let budget = PagedKvCache::new(self.kv_config, self.kv_total_blocks);
        let mut orchestrator = ForwardOrchestrator::resume(&self.kernel, budget, snapshot)?;
        let mut hidden = vec![0.0f32; self.hidden_size];
        for &token in suffix {
            hidden = self.embed(token)?;
            orchestrator.decode_token(&mut hidden)?;
        }
        Ok(GenerationSession {
            generator: self,
            orchestrator,
            last_hidden: hidden,
            rng: SplitMix64::new(sampler.seed()),
            sampler,
        })
    }
}

impl<K: ComputeKernel> GenerationSession<'_, K> {
    /// Snapshot this session's KV history (e.g. right after prefilling a prompt)
    /// so a later prompt sharing this prefix can
    /// [`resume`](Generator::resume_session) from it.
    pub fn snapshot(&self) -> crate::forward::KvSnapshot {
        self.orchestrator.snapshot()
    }

    /// Emit the next token and advance the internal state by one step.
    pub fn step(&mut self) -> Result<u32> {
        let mut logits = self.generator.logits(&self.last_hidden);
        apply_repetition_penalty(&mut logits, &self.seen, self.sampler.repetition_penalty());
        let next = self.sampler.sample(&logits, &mut self.rng);
        self.seen.insert(next);
        self.last_hidden = self.generator.embed(next)?;
        self.orchestrator.decode_token(&mut self.last_hidden)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{BlockConfig, CpuKernel, LayerTensors};

    /// A tiny deterministic model: identity transformer block, one-hot
    /// embedding, and an LM head that shifts the argmax by +1 (mod vocab), so
    /// generation counts upward from the prompt token.
    fn counting_generator() -> Generator<CpuKernel> {
        let vocab = 4usize;
        let hidden = 4usize;
        let cfg = BlockConfig {
            hidden_size: hidden,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        };
        // One identity (zero-weight) block: hidden passes through unchanged.
        let kernel = CpuKernel::new(cfg, vec![LayerTensors::zeros(&cfg)]).unwrap();

        // Embedding = identity: token t → one-hot e_t.
        let mut embedding = vec![0.0f32; vocab * hidden];
        for t in 0..vocab {
            embedding[t * hidden + t] = 1.0;
        }
        // LM head row j = one-hot at (j-1) mod vocab, so the row aligned with the
        // normed one-hot (peaking at t) is row (t+1) → argmax = (t+1) mod vocab.
        let mut lm_head = vec![0.0f32; vocab * hidden];
        for j in 0..vocab {
            lm_head[j * hidden + (j + vocab - 1) % vocab] = 1.0;
        }
        let final_norm = vec![1.0f32; hidden];

        let kv_config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 2,
            block_size: 16,
        };
        Generator::new(kernel, embedding, final_norm, lm_head, vocab, 1e-5, kv_config, 8).unwrap()
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[5.0, 5.0, 1.0]), 0); // first max wins
    }

    #[test]
    fn samplers_behave() {
        let logits = [1.0f32, 5.0, 2.0, 0.5];
        let mut rng = SplitMix64::new(42);

        // temperature 0 collapses to greedy.
        let zero = Sampler::TopPK { temperature: 0.0, top_p: 1.0, top_k: 0, min_p: 0.0, repetition_penalty: 1.0, seed: 1 };
        assert_eq!(zero.sample(&logits, &mut rng), 1);

        // top_k = 1 keeps only the argmax, so any draw returns it.
        let k1 = Sampler::TopPK { temperature: 2.0, top_p: 1.0, top_k: 1, min_p: 0.0, repetition_penalty: 1.0, seed: 7 };
        for _ in 0..20 {
            assert_eq!(k1.sample(&logits, &mut rng), 1);
        }

        // A dominant logit + tight nucleus keeps only that token.
        let peaked = [0.0f32, 0.0, 20.0, 0.0];
        let nucleus = Sampler::TopPK { temperature: 1.0, top_p: 0.5, top_k: 0, min_p: 0.0, repetition_penalty: 1.0, seed: 3 };
        for _ in 0..20 {
            assert_eq!(nucleus.sample(&peaked, &mut rng), 2);
        }

        // A fixed seed makes temperature sampling reproducible.
        let s = Sampler::TopPK { temperature: 1.5, top_p: 1.0, top_k: 0, min_p: 0.0, repetition_penalty: 1.0, seed: 99 };
        let mut a = SplitMix64::new(s.seed());
        let mut b = SplitMix64::new(s.seed());
        let seq_a: Vec<u32> = (0..8).map(|_| s.sample(&logits, &mut a)).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| s.sample(&logits, &mut b)).collect();
        assert_eq!(seq_a, seq_b);

        // min_p prunes tokens far below the top probability. With one dominant
        // logit and a high min_p, only the top token survives.
        let peaked = [0.0f32, 5.0, 0.0, 0.0];
        let mp = Sampler::TopPK { temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.5, repetition_penalty: 1.0, seed: 4 };
        for _ in 0..20 {
            assert_eq!(mp.sample(&peaked, &mut rng), 1);
        }
        // min_p = 0 disables the filter (flat logits → draws vary across tokens).
        let off = Sampler::TopPK { temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.0, repetition_penalty: 1.0, seed: 4 };
        let flat = [1.0f32, 1.0, 1.0, 1.0];
        let draws: std::collections::HashSet<u32> =
            (0..50).map(|_| off.sample(&flat, &mut rng)).collect();
        assert!(draws.len() > 1, "min_p=0 should not collapse a flat distribution");
    }

    #[test]
    fn repetition_penalty_reweights_seen_tokens() {
        use std::collections::HashSet;
        let seen: HashSet<u32> = [0u32, 2].into_iter().collect();

        // Positive logits of seen tokens are divided; negative ones multiplied
        // (pushed further down). Unseen tokens (1, 3) are untouched.
        let mut logits = vec![2.0f32, 2.0, -2.0, -2.0];
        apply_repetition_penalty(&mut logits, &seen, 2.0);
        assert_eq!(logits, vec![1.0, 2.0, -4.0, -2.0]);

        // A penalty of 1.0 is a no-op.
        let mut same = vec![3.0f32, -1.0, 0.5];
        apply_repetition_penalty(&mut same, &seen, 1.0);
        assert_eq!(same, vec![3.0, -1.0, 0.5]);

        // End-to-end: penalizing the counting model's next token diverts it.
        // Unpenalized, [0] → 1; a strong penalty on token 1 (once seen) reshapes
        // later steps, so the two sequences differ.
        let gen = counting_generator();
        let cfg = |rp: f32| GenerationConfig {
            max_new_tokens: 6,
            eos_token: None,
            sampler: Sampler::TopPK {
                temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.0, repetition_penalty: rp, seed: 1,
            },
        };
        let base = gen.generate(&[0], &cfg(1.0)).unwrap();
        let penalized = gen.generate(&[0], &cfg(10.0)).unwrap();
        assert_ne!(base, penalized, "repetition_penalty should change the sequence");
    }

    /// A small random-weight generator with real attention, so output depends
    /// on the KV history (unlike the zero-weight counting model).
    fn attention_generator() -> Generator<CpuKernel> {
        let (vocab, hidden) = (16usize, 16usize);
        let cfg = BlockConfig {
            hidden_size: hidden,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        };
        let mut r = SplitMix64::new(42);
        let mut vec = |n: usize| -> Vec<f32> {
            (0..n).map(|_| r.next_f32() * 0.1 - 0.05).collect()
        };
        let layers = vec![LayerTensors {
            q_proj: vec(cfg.q_dim() * hidden),
            k_proj: vec(cfg.kv_dim() * hidden),
            v_proj: vec(cfg.kv_dim() * hidden),
            o_proj: vec(hidden * cfg.q_dim()),
            gate_proj: vec(cfg.intermediate_size * hidden),
            up_proj: vec(cfg.intermediate_size * hidden),
            down_proj: vec(hidden * cfg.intermediate_size),
            input_layernorm: std::vec::from_elem(1.0, hidden),
            post_attention_layernorm: std::vec::from_elem(1.0, hidden),
        }];
        let kernel = CpuKernel::new(cfg, layers).unwrap();
        Generator::new(
            kernel,
            vec(vocab * hidden),
            std::vec::from_elem(1.0, hidden),
            vec(vocab * hidden),
            vocab,
            1e-5,
            KvCacheConfig { num_layers: 1, num_kv_heads: 2, head_dim: 4, block_size: 16 },
            64,
        )
        .unwrap()
    }

    #[test]
    fn resume_from_snapshot_matches_full_prefill() {
        let gen = attention_generator();
        let prefix = [1u32, 2, 3];
        let suffix = [4u32, 5];
        let full_prompt = [1u32, 2, 3, 4, 5];
        let n = 6;

        // Full prefill of the whole prompt.
        let mut full = gen.start_session(&full_prompt, Sampler::Greedy).unwrap();
        let tokens_full: Vec<u32> = (0..n).map(|_| full.step().unwrap()).collect();

        // Snapshot after the prefix, resume prefilling only the suffix.
        let prefix_sess = gen.start_session(&prefix, Sampler::Greedy).unwrap();
        let snap = prefix_sess.snapshot();
        assert_eq!(snap.position(), prefix.len());
        let mut resumed = gen.resume_session(snap, &suffix, Sampler::Greedy).unwrap();
        let tokens_resumed: Vec<u32> = (0..n).map(|_| resumed.step().unwrap()).collect();

        // Resuming from the shared prefix is bit-for-bit identical to prefilling
        // the full prompt — the correctness property a prefix cache relies on.
        assert_eq!(tokens_full, tokens_resumed);
    }

    #[test]
    fn generates_counting_sequence() {
        let gen = counting_generator();
        let out = gen
            .generate(
                &[0],
                &GenerationConfig {
                    max_new_tokens: 3,
                    eos_token: None,
                    sampler: Sampler::Greedy,
                },
            )
            .unwrap();
        // From token 0, counts up: 1, 2, 3.
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn stops_at_eos() {
        let gen = counting_generator();
        let out = gen
            .generate(
                &[0],
                &GenerationConfig {
                    max_new_tokens: 10,
                    eos_token: Some(2),
                    sampler: Sampler::Greedy,
                },
            )
            .unwrap();
        // 1, then 2 (== eos) → stops, eos included.
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn rejects_empty_prompt_and_out_of_range_token() {
        let gen = counting_generator();
        assert!(gen.generate(&[], &GenerationConfig::default()).is_err());
        assert!(gen.generate(&[99], &GenerationConfig::default()).is_err());
    }
}
