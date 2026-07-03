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
use crate::error::{FlipError, Result};
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

/// Token-selection strategy over a logit vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// Deterministic argmax.
    Greedy,
}

impl Sampler {
    /// Pick the next token id from `logits`.
    pub fn sample(&self, logits: &[f32]) -> u32 {
        match self {
            Sampler::Greedy => argmax(logits),
        }
    }
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
                return Err(FlipError::InvalidConfig(format!(
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
            return Err(FlipError::InvalidConfig(format!(
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
            return Err(FlipError::InvalidConfig("prompt must be non-empty".into()));
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

        // Decode loop.
        let mut generated = Vec::with_capacity(cfg.max_new_tokens);
        for _ in 0..cfg.max_new_tokens {
            let logits = self.logits(&hidden);
            let next = cfg.sampler.sample(&logits);
            generated.push(next);
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
}

impl<K: ComputeKernel> Generator<K> {
    /// Begin a step-wise generation: prefills `prompt` and leaves the session
    /// ready to emit the first continuation token via [`GenerationSession::step`].
    pub fn start_session(&self, prompt: &[u32], sampler: Sampler) -> Result<GenerationSession<'_, K>> {
        if prompt.is_empty() {
            return Err(FlipError::InvalidConfig("prompt must be non-empty".into()));
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
            sampler,
        })
    }
}

impl<K: ComputeKernel> GenerationSession<'_, K> {
    /// Emit the next token and advance the internal state by one step.
    pub fn step(&mut self) -> Result<u32> {
        let logits = self.generator.logits(&self.last_hidden);
        let next = self.sampler.sample(&logits);
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
