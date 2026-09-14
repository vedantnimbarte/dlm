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
use crate::forward::cpu::{matvec, KvLayerCache};
use crate::forward::{ComputeKernel, ForwardOrchestrator};

/// Prompt tokens staged per prefill call. Bounds the `tokens × hidden` buffer on
/// a long prompt; a streaming kernel loads each layer once per chunk.
const PREFILL_CHUNK: usize = 512;

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
            Sampler::TopPK {
                repetition_penalty, ..
            } => *repetition_penalty,
        }
    }

    /// The distribution [`sample`](Self::sample) draws from, as `(token, prob)`
    /// pairs over the tokens that survive truncation, most likely first. Greedy
    /// (or `temperature <= 0`) is the one-hot distribution on the argmax.
    /// Speculative decoding needs the distributions themselves, not a draw.
    pub fn distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        match *self {
            Sampler::TopPK {
                temperature,
                top_p,
                top_k,
                min_p,
                ..
            } if temperature > 0.0 && !logits.is_empty() => {
                let (idx, probs) =
                    topk_topp_distribution(logits, temperature, top_p, top_k as usize, min_p);
                idx.into_iter().map(|i| i as u32).zip(probs).collect()
            }
            _ => vec![(argmax(logits), 1.0)],
        }
    }

    /// Pick the next token id from `logits`, advancing `rng` for stochastic
    /// samplers (unused by [`Greedy`](Sampler::Greedy)).
    pub fn sample(&self, logits: &[f32], rng: &mut SplitMix64) -> u32 {
        match *self {
            Sampler::Greedy => argmax(logits),
            Sampler::TopPK {
                temperature,
                top_p,
                top_k,
                min_p,
                ..
            } => {
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
fn apply_repetition_penalty(
    logits: &mut [f32],
    seen: &std::collections::HashSet<u32>,
    penalty: f32,
) {
    if penalty == 1.0 {
        return;
    }
    for &t in seen {
        let i = t as usize;
        if i < logits.len() {
            logits[i] = if logits[i] > 0.0 {
                logits[i] / penalty
            } else {
                logits[i] * penalty
            };
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
    let (idx, probs) = topk_topp_distribution(logits, temperature, top_p, top_k, min_p);
    let pairs: Vec<(u32, f32)> = idx.into_iter().map(|i| i as u32).zip(probs).collect();
    sample_from(&pairs, rng)
}

/// Inverse-CDF draw from `(token, prob)` pairs, in the order given. Falls back
/// to the last token when rounding leaves the cumulative mass just short of 1.
pub(crate) fn sample_from(dist: &[(u32, f32)], rng: &mut SplitMix64) -> u32 {
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for &(token, p) in dist {
        cum += p;
        if r < cum {
            return token;
        }
    }
    dist.last().map_or(0, |&(token, _)| token)
}

/// The filtered, renormalized distribution behind [`sample_topk_topp`]: token
/// indices sorted by logit (descending) and their probabilities. `logits` must
/// be non-empty.
fn topk_topp_distribution(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
) -> (Vec<usize>, Vec<f32>) {
    // Candidates by logit, descending, ties broken by token id so a seed always
    // draws the same token. Only a prefix of that order is ever used, so select
    // the prefix before sorting instead of sorting the whole vocabulary (152k
    // entries for Qwen, every sampled token).
    let desc = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    let top = |k: usize| -> Vec<(f32, u32)> {
        let mut c: Vec<(f32, u32)> = logits.iter().copied().zip(0u32..).collect();
        if k < c.len() {
            c.select_nth_unstable_by(k - 1, desc);
            c.truncate(k);
        }
        c.sort_unstable_by(desc);
        c
    };
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weight = |l: f32| ((l - max_logit) / temperature).exp();

    let cand = if top_k > 0 {
        top(top_k)
    } else if top_p > 0.0 && top_p < 1.0 {
        // The nucleus is the shortest prefix holding `top_p` of the mass. Grow
        // the selected prefix until it holds that much.
        let total: f32 = logits.iter().map(|&l| weight(l)).sum();
        let mut k = 256;
        loop {
            let c = top(k.min(logits.len()));
            let mass: f32 = c.iter().map(|&(l, _)| weight(l)).sum();
            if mass >= top_p * total || k >= logits.len() {
                break c;
            }
            k *= 4;
        }
    } else {
        top(logits.len())
    };
    let mut idx: Vec<usize> = cand.iter().map(|&(_, i)| i as usize).collect();

    // Softmax with temperature. Top-k renormalizes over the tokens it keeps;
    // otherwise probabilities are over the whole vocabulary.
    let mut probs: Vec<f32> = cand.iter().map(|&(l, _)| weight(l)).collect();
    let sum: f32 = if top_k > 0 {
        probs.iter().sum()
    } else {
        logits.iter().map(|&l| weight(l)).sum()
    };
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

    (idx, probs)
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
    /// Per-layer KV precision (int8/int4 shrink KV memory, approximate).
    kv_quant: crate::forward::KvQuant,
    /// Scalar applied to each token embedding after lookup (Gemma multiplies by
    /// `sqrt(hidden)`); `None` leaves embeddings unscaled.
    embed_scale: Option<f32>,
    /// GPT-2's learned absolute position embeddings (`wpe`), `[max_pos, hidden]`.
    ///
    /// GPT-2 encodes position by *adding* this to the token embedding before the
    /// first block, and applies no rotary at all. Leaving it out is not a subtle
    /// loss of quality -- the model then has no positional signal whatsoever and
    /// degenerates into repeating one token.
    position_embedding: Option<Vec<f32>>,
    /// Bias for the final norm. Present only when that norm is a LayerNorm.
    final_norm_bias: Option<Vec<f32>>,
    /// Which normalization the final norm uses; must match the blocks'.
    final_norm_kind: crate::forward::cpu::NormKind,
    /// Gemma2 caps the final logits at `tanh(l/cap)*cap` before sampling, which
    /// changes the distribution (it compresses the tail toward the cap), so it is
    /// part of correctness rather than a stylistic knob. `None` elsewhere.
    final_logit_softcap: Option<f32>,
    /// The LM head on the device, when [`with_gpu_lm_head`](Self::with_gpu_lm_head)
    /// placed it there; `lm_head` is then left empty.
    #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
    gpu_head: Option<crate::forward::GpuLmHead>,
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
            kv_quant: crate::forward::KvQuant::None,
            embed_scale: None,
            position_embedding: None,
            final_norm_bias: None,
            final_norm_kind: crate::forward::cpu::NormKind::Rms,
            final_logit_softcap: None,
            #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
            gpu_head: None,
        })
    }

    /// Set the per-layer KV precision (int8/int4 shrink KV memory, approximate).
    /// Affects sessions started after this call.
    pub fn with_kv_quant(mut self, kv_quant: crate::forward::KvQuant) -> Self {
        self.kv_quant = kv_quant;
        self
    }

    /// Scale token embeddings by `scale` after lookup (Gemma uses `sqrt(hidden)`).
    /// `None` leaves them unscaled.
    /// Attach learned position embeddings and the final-norm kind.
    ///
    /// `norm_kind` is passed explicitly rather than inferred from the presence of
    /// `position_embedding`: Falcon needs a LayerNorm head and has no `wpe`, so
    /// inferring one from the other would silently give it an RMSNorm head.
    pub fn with_head(
        mut self,
        position_embedding: Option<Vec<f32>>,
        final_norm_bias: Option<Vec<f32>>,
        norm_kind: crate::forward::cpu::NormKind,
    ) -> Self {
        self.final_norm_kind = norm_kind;
        self.position_embedding = position_embedding;
        self.final_norm_bias = final_norm_bias;
        self
    }

    pub fn with_embed_scale(mut self, scale: Option<f32>) -> Self {
        self.embed_scale = scale;
        self
    }

    /// Cap the final logits at `tanh(l/cap)*cap` before sampling (Gemma2).
    pub fn with_final_logit_softcap(mut self, cap: Option<f32>) -> Self {
        self.final_logit_softcap = cap.filter(|c| *c > 0.0);
        self
    }

    /// Shorthand for int8 KV (about half the memory).
    pub fn with_quantized_kv(self) -> Self {
        self.with_kv_quant(crate::forward::KvQuant::Int8)
    }

    /// Layer-streaming cache stats, if the kernel streams weights (else `None`).
    pub fn stream_stats(&self) -> Option<crate::forward::StreamStats> {
        self.kernel.stream_stats()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embed a token id into a fresh hidden vector.
    /// Move the LM head into VRAM, so each step's vocabulary-wide GEMV runs on the
    /// device instead of the host.
    ///
    /// Precision follows the weights: a head that is exactly bf16-representable
    /// (a bf16 checkpoint) uploads as bf16, losslessly and at half the size;
    /// anything else uploads as f32. `quantize_int8` is for a model whose layers
    /// were quantized at load (`--quant int4`/`int8`) -- the user already chose
    /// lossy weights, and int8 keeps the head's error small (int4 is too coarse
    /// for the one matrix every token's choice goes through).
    ///
    /// `device` names the GPU to hold it; `None` uses the current device, which
    /// is right for a single-GPU kernel. A multi-GPU pipeline passes its last
    /// stage's device, since no one device is current when logits are computed.
    ///
    /// The host copy is released once the upload succeeds; on failure (VRAM
    /// exhausted, say) the generator is untouched and keeps the host head.
    #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
    pub fn place_lm_head_on_gpu(&mut self, quantize_int8: bool, device: Option<u32>) -> Result<()> {
        use crate::forward::Weights;
        let head = if quantize_int8 {
            Weights::quantize_int8(&self.lm_head, crate::forward::QUANT_GROUP_SIZE)?
        } else if self.lm_head.iter().all(|v| v.to_bits() & 0xFFFF == 0) {
            Weights::Bf16(
                self.lm_head
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect(),
            )
        } else {
            Weights::from_f32(self.lm_head.clone())
        };
        self.gpu_head = Some(crate::forward::GpuLmHead::new(
            &head,
            self.vocab_size,
            self.hidden_size,
            device,
        )?);
        self.lm_head = Vec::new();
        Ok(())
    }

    /// [`place_lm_head_on_gpu`](Self::place_lm_head_on_gpu), falling back to the
    /// host head with a warning rather than failing: a host head is slower, not
    /// wrong.
    #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
    pub fn with_lm_head_on_gpu(mut self, quantize_int8: bool, device: Option<u32>) -> Self {
        if let Err(e) = self.place_lm_head_on_gpu(quantize_int8, device) {
            eprintln!("warning: the LM head stays on the CPU ({e}); decoding will be slower");
        }
        self
    }

    /// A fresh single-sequence orchestrator over this generator's kernel and KV
    /// budget.
    pub(crate) fn orchestrator(&self) -> ForwardOrchestrator<&K> {
        ForwardOrchestrator::new(
            &self.kernel,
            PagedKvCache::new(self.kv_config, self.kv_total_blocks),
            self.kv_quant,
        )
    }

    /// The sampler's next-token distribution after `hidden`, with the repetition
    /// penalty applied over `seen` -- exactly what a session samples from.
    pub(crate) fn distribution(
        &self,
        hidden: &[f32],
        seen: &std::collections::HashSet<u32>,
        sampler: &Sampler,
    ) -> Result<Vec<(u32, f32)>> {
        let mut logits = self.logits(hidden)?;
        apply_repetition_penalty(&mut logits, seen, sampler.repetition_penalty());
        Ok(sampler.distribution(&logits))
    }

    pub(crate) fn embed(&self, token: u32, position: usize) -> Result<Vec<f32>> {
        let idx = token as usize;
        if idx >= self.vocab_size {
            return Err(DlmError::InvalidConfig(format!(
                "token {token} out of vocab range {}",
                self.vocab_size
            )));
        }
        let start = idx * self.hidden_size;
        let mut v = self.embedding[start..start + self.hidden_size].to_vec();
        if let Some(scale) = self.embed_scale {
            for x in &mut v {
                *x *= scale;
            }
        }
        // GPT-2: add the learned position embedding. Clamped rather than wrapped
        // past the trained range -- a wrapped position is silently wrong, while a
        // clamped one degrades predictably at the edge.
        if let Some(wpe) = &self.position_embedding {
            let rows = wpe.len() / self.hidden_size;
            let row = position.min(rows.saturating_sub(1));
            let base = row * self.hidden_size;
            for (x, p) in v.iter_mut().zip(&wpe[base..base + self.hidden_size]) {
                *x += p;
            }
        }
        Ok(v)
    }

    /// Run `tokens` (non-empty) through the model as one prefill, in chunks of
    /// [`PREFILL_CHUNK`] so a long prompt's staged hidden states stay bounded.
    /// Returns the last token's hidden state.
    pub(crate) fn prefill(
        &self,
        orch: &mut ForwardOrchestrator<&K>,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let mut hiddens = Vec::new();
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            hiddens.clear();
            for (i, &token) in chunk.iter().enumerate() {
                hiddens.extend(self.embed(token, orch.position() + i)?);
            }
            orch.prefill(&mut hiddens)?;
        }
        Ok(hiddens.split_off(hiddens.len() - self.hidden_size))
    }

    /// The log-probability the model assigns each token given the ones before it
    /// (teacher forcing): `out[i]` is `log p(tokens[i + 1] | tokens[..=i])`, so
    /// the result is one shorter than `tokens`. The mean of its negation is the
    /// text's cross-entropy -- the number that tells a subtly wrong forward pass
    /// (a misplaced norm, a wrong RoPE base) from a right one when greedy output
    /// still looks fine.
    pub fn score(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut orch = self.orchestrator();
        let mut out = Vec::with_capacity(tokens.len().saturating_sub(1));
        for (c, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            let mut hiddens = Vec::with_capacity(chunk.len() * self.hidden_size);
            for (i, &token) in chunk.iter().enumerate() {
                hiddens.extend(self.embed(token, orch.position() + i)?);
            }
            orch.prefill(&mut hiddens)?;
            for (i, hidden) in hiddens.chunks(self.hidden_size).enumerate() {
                let Some(&next) = tokens.get(c * PREFILL_CHUNK + i + 1) else {
                    break;
                };
                let logits = self.logits(hidden)?;
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = max
                    + logits
                        .iter()
                        .map(|&l| ((l - max) as f64).exp())
                        .sum::<f64>()
                        .ln() as f32;
                out.push(logits[next as usize] - lse);
            }
        }
        Ok(out)
    }

    /// Project a hidden state to vocabulary logits via final norm + LM head.
    fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let normed = crate::forward::cpu::norm(
            hidden,
            &self.final_norm,
            self.final_norm_bias.as_deref(),
            self.rms_eps,
            self.final_norm_kind,
        );
        #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
        let mut out = match &self.gpu_head {
            Some(head) => head.logits(&normed)?,
            None => matvec(&self.lm_head, &normed, self.vocab_size, self.hidden_size),
        };
        #[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
        let mut out = matvec(&self.lm_head, &normed, self.vocab_size, self.hidden_size);
        if let Some(cap) = self.final_logit_softcap {
            for l in out.iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }
        Ok(out)
    }

    /// Greedy-decode a **batch** of prompts together, advancing all sequences
    /// through each layer in one [`run_block_batched`](ComputeKernel::run_block_batched)
    /// call — so a GPU kernel fuses the per-sequence projections into batched
    /// GEMMs. Each sequence's output is identical to decoding it alone (the batch
    /// is a throughput optimization, not a semantic change).
    ///
    /// Prompts may differ in length (each carries its own position and KV). A
    /// sequence that hits EOS stops emitting but still rides the batch until the
    /// longest finishes (a scheduler would retire it; this is the simple form).
    pub fn generate_batch(
        &self,
        prompts: &[&[u32]],
        cfg: &GenerationConfig,
    ) -> Result<Vec<Vec<u32>>> {
        let b = prompts.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        for p in prompts {
            if p.is_empty() {
                return Err(DlmError::InvalidConfig("prompt must be non-empty".into()));
            }
        }
        let nl = self.kernel.num_layers() as usize;
        let kv_dim = self.kernel.kv_dim();
        // Per-sequence KV (one cache per layer) + hidden + absolute position.
        let mut kvs: Vec<Vec<KvLayerCache>> = (0..b)
            .map(|_| {
                (0..nl)
                    .map(|_| KvLayerCache::new_quant(kv_dim, self.kv_quant))
                    .collect()
            })
            .collect();
        let mut hidden = vec![vec![0.0f32; self.hidden_size]; b];
        let mut position = vec![0usize; b];

        // Prefill each sequence (per-sequence — prompts differ in length). After
        // the loop `hidden[s]` holds the last prompt token's stack output (what the
        // first decode step samples from) and `position[s]` is the next position.
        for (s, prompt) in prompts.iter().enumerate() {
            for &tok in *prompt {
                // Attribute layer events to a step, so a consumer can group a
                // token's worth of transfers together. Prefill counts as step 0:
                // it is one pass over the stack per prompt token, but it is the
                // decode steps that show the steady-state streaming cost.
                crate::telemetry::set_token(0);
                let mut h = self.embed(tok, position[s])?;
                for (l, kv) in kvs[s].iter_mut().enumerate() {
                    self.kernel.run_block(l as u32, &mut h, kv, position[s])?;
                }
                hidden[s] = h;
                position[s] += 1;
            }
        }

        let mut rngs: Vec<SplitMix64> = (0..b)
            .map(|_| SplitMix64::new(cfg.sampler.seed()))
            .collect();
        let penalty = cfg.sampler.repetition_penalty();
        let mut seen: Vec<std::collections::HashSet<u32>> = prompts
            .iter()
            .map(|p| p.iter().copied().collect())
            .collect();
        let mut out = vec![Vec::with_capacity(cfg.max_new_tokens); b];
        let mut done = vec![false; b];

        for step in 0..cfg.max_new_tokens {
            crate::telemetry::set_token(step as u64 + 1);
            // Sample the next token for each live sequence from its current hidden.
            for s in 0..b {
                if done[s] {
                    continue;
                }
                let mut logits = self.logits(&hidden[s])?;
                apply_repetition_penalty(&mut logits, &seen[s], penalty);
                let next = cfg.sampler.sample(&logits, &mut rngs[s]);
                out[s].push(next);
                seen[s].insert(next);
                if cfg.eos_token == Some(next) {
                    done[s] = true;
                    continue;
                }
                hidden[s] = self.embed(next, position[s])?;
            }
            if done.iter().all(|&d| d) {
                break;
            }
            // Advance every sequence through the stack, one batched call per layer.
            for l in 0..nl {
                let mut hs: Vec<&mut [f32]> = hidden.iter_mut().map(|h| h.as_mut_slice()).collect();
                let mut ks: Vec<&mut KvLayerCache> =
                    kvs.iter_mut().map(|seq| &mut seq[l]).collect();
                self.kernel
                    .run_block_batched(l as u32, &mut hs, &mut ks, &position)?;
            }
            for p in &mut position {
                *p += 1;
            }
        }
        Ok(out)
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
        let mut orch = ForwardOrchestrator::new(&self.kernel, budget, self.kv_quant);

        // Prefill: run every prompt token, carrying the last hidden state.
        let mut hidden = self.prefill(&mut orch, prompt)?;

        // Decode loop. `seen` tracks the full context (prompt + generated) for
        // the repetition penalty.
        let mut rng = SplitMix64::new(cfg.sampler.seed());
        let penalty = cfg.sampler.repetition_penalty();
        let mut seen: std::collections::HashSet<u32> = prompt.iter().copied().collect();
        let mut generated = Vec::with_capacity(cfg.max_new_tokens);
        for _ in 0..cfg.max_new_tokens {
            let mut logits = self.logits(&hidden)?;
            apply_repetition_penalty(&mut logits, &seen, penalty);
            let next = cfg.sampler.sample(&logits, &mut rng);
            generated.push(next);
            seen.insert(next);
            if cfg.eos_token == Some(next) {
                break;
            }
            hidden = self.embed(next, orch.position())?;
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
    pub fn start_session(
        &self,
        prompt: &[u32],
        sampler: Sampler,
    ) -> Result<GenerationSession<'_, K>> {
        if prompt.is_empty() {
            return Err(DlmError::InvalidConfig("prompt must be non-empty".into()));
        }
        let budget = crate::cache::PagedKvCache::new(self.kv_config, self.kv_total_blocks);
        let mut orchestrator =
            crate::forward::ForwardOrchestrator::new(&self.kernel, budget, self.kv_quant);
        let hidden = self.prefill(&mut orchestrator, prompt)?;
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
    /// the tokens of `prompt` that follow the snapshotted prefix. The result is
    /// identical to [`start_session`](Self::start_session) on the full `prompt`,
    /// but skips re-running the shared prefix — the basis for cross-request prefix
    /// caching.
    ///
    /// `prompt` is the **whole** prompt (the cached prefix plus the new suffix);
    /// the snapshot's position marks where the prefix ends. Passing the full
    /// prompt lets the repetition penalty cover the cached prefix's tokens too —
    /// otherwise a resumed request would penalize only its suffix and drift from
    /// the same request run without the cache. The suffix (`prompt[position..]`)
    /// must be non-empty (the session needs a last hidden state).
    pub fn resume_session(
        &self,
        snapshot: crate::forward::KvSnapshot,
        prompt: &[u32],
        sampler: Sampler,
    ) -> Result<GenerationSession<'_, K>> {
        let start = snapshot.position();
        if start > prompt.len() {
            return Err(DlmError::InvalidConfig(
                "resume snapshot is longer than the prompt it should prefix".into(),
            ));
        }
        let suffix = &prompt[start..];
        if suffix.is_empty() {
            return Err(DlmError::InvalidConfig(
                "resume suffix must be non-empty".into(),
            ));
        }
        let budget = PagedKvCache::new(self.kv_config, self.kv_total_blocks);
        let mut orchestrator = ForwardOrchestrator::resume(&self.kernel, budget, snapshot)?;
        let hidden = self.prefill(&mut orchestrator, suffix)?;
        Ok(GenerationSession {
            generator: self,
            orchestrator,
            last_hidden: hidden,
            rng: SplitMix64::new(sampler.seed()),
            sampler,
            // The full prompt (cached prefix + suffix), so the repetition penalty
            // covers the cached prefix — a resumed request matches the same request
            // run without the prefix cache.
            seen: prompt.iter().copied().collect(),
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

    /// Snapshot for the prefix cache, pulling device-resident K/V back to the
    /// host first so the snapshot is real on the GPU kernels too (where the host
    /// caches otherwise hold only length placeholders). See
    /// [`ForwardOrchestrator::snapshot_synced`](crate::forward::ForwardOrchestrator::snapshot_synced).
    pub fn snapshot_synced(&mut self) -> Result<crate::forward::KvSnapshot> {
        self.orchestrator.snapshot_synced()
    }

    /// Emit the next token and advance the internal state by one step.
    pub fn step(&mut self) -> Result<u32> {
        let mut logits = self.generator.logits(&self.last_hidden)?;
        apply_repetition_penalty(&mut logits, &self.seen, self.sampler.repetition_penalty());
        let next = self.sampler.sample(&logits, &mut self.rng);
        self.seen.insert(next);
        self.last_hidden = self.generator.embed(next, self.orchestrator.position())?;
        self.orchestrator.decode_token(&mut self.last_hidden)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::Weights;
    use crate::forward::{BlockConfig, CpuKernel, ExpertFfn, Ffn, LayerTensors};

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
            rope_scaling: None,
            moe: None,
            sliding_window: None,
            activation: Default::default(),
            mla: None,
            ..Default::default()
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
        Generator::new(
            kernel, embedding, final_norm, lm_head, vocab, 1e-5, kv_config, 8,
        )
        .unwrap()
    }

    /// `score` must be the log-softmax of the next token, computed the slow way:
    /// one decode step at a time, full logits, explicit normalization.
    #[test]
    fn score_is_the_next_tokens_log_softmax() {
        let g = counting_generator();
        let tokens = [0u32, 1, 3, 2, 2];
        let got = g.score(&tokens).unwrap();
        assert_eq!(got.len(), tokens.len() - 1);
        let mut orch = g.orchestrator();
        for (i, &t) in tokens[..tokens.len() - 1].iter().enumerate() {
            let mut h = g.embed(t, i).unwrap();
            orch.decode_token(&mut h).unwrap();
            let logits = g.logits(&h).unwrap();
            let z: f32 = logits.iter().map(|l| l.exp()).sum();
            let want = logits[tokens[i + 1] as usize] - z.ln();
            assert!(
                (got[i] - want).abs() < 1e-5,
                "token {i}: {} vs {want}",
                got[i]
            );
        }
    }

    /// Gemma2's final-logit softcap squashes logits through `tanh(l/cap)*cap`
    /// before sampling. It must compress the spread (that is the whole point —
    /// it changes the sampled distribution) while leaving the ranking intact,
    /// since `tanh` is monotonic.
    #[test]
    fn final_logit_softcap_compresses_without_reordering() {
        let hidden = vec![0.4f32, -0.7, 0.15, 0.9];
        let plain = counting_generator();
        let raw = plain.logits(&hidden).unwrap();

        let capped_gen = counting_generator().with_final_logit_softcap(Some(0.5));
        let capped = capped_gen.logits(&hidden).unwrap();

        assert_eq!(raw.len(), capped.len());
        // Every capped logit is inside ±cap, and strictly smaller in magnitude
        // wherever the raw logit had any real magnitude.
        for (r, c) in raw.iter().zip(&capped) {
            assert!(c.abs() <= 0.5 + 1e-6, "logit {c} escaped the cap");
            if r.abs() > 1e-3 {
                assert!(c.abs() < r.abs(), "cap should shrink {r} but gave {c}");
            }
        }
        // Monotonic ⇒ argmax (and the whole ranking) is unchanged.
        assert_eq!(argmax(&raw), argmax(&capped));

        // A non-positive cap is treated as "off", not as a divide-by-zero.
        let off = counting_generator().with_final_logit_softcap(Some(0.0));
        assert_eq!(off.logits(&hidden).unwrap(), raw);
        assert_eq!(
            counting_generator()
                .with_final_logit_softcap(None)
                .logits(&hidden)
                .unwrap(),
            raw
        );
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
        let zero = Sampler::TopPK {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 1,
        };
        assert_eq!(zero.sample(&logits, &mut rng), 1);

        // top_k = 1 keeps only the argmax, so any draw returns it.
        let k1 = Sampler::TopPK {
            temperature: 2.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 7,
        };
        for _ in 0..20 {
            assert_eq!(k1.sample(&logits, &mut rng), 1);
        }

        // A dominant logit + tight nucleus keeps only that token.
        let peaked = [0.0f32, 0.0, 20.0, 0.0];
        let nucleus = Sampler::TopPK {
            temperature: 1.0,
            top_p: 0.5,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 3,
        };
        for _ in 0..20 {
            assert_eq!(nucleus.sample(&peaked, &mut rng), 2);
        }

        // A fixed seed makes temperature sampling reproducible.
        let s = Sampler::TopPK {
            temperature: 1.5,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 99,
        };
        let mut a = SplitMix64::new(s.seed());
        let mut b = SplitMix64::new(s.seed());
        let seq_a: Vec<u32> = (0..8).map(|_| s.sample(&logits, &mut a)).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| s.sample(&logits, &mut b)).collect();
        assert_eq!(seq_a, seq_b);

        // min_p prunes tokens far below the top probability. With one dominant
        // logit and a high min_p, only the top token survives.
        let peaked = [0.0f32, 5.0, 0.0, 0.0];
        let mp = Sampler::TopPK {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.5,
            repetition_penalty: 1.0,
            seed: 4,
        };
        for _ in 0..20 {
            assert_eq!(mp.sample(&peaked, &mut rng), 1);
        }
        // min_p = 0 disables the filter (flat logits → draws vary across tokens).
        let off = Sampler::TopPK {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 4,
        };
        let flat = [1.0f32, 1.0, 1.0, 1.0];
        let draws: std::collections::HashSet<u32> =
            (0..50).map(|_| off.sample(&flat, &mut rng)).collect();
        assert!(
            draws.len() > 1,
            "min_p=0 should not collapse a flat distribution"
        );
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
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                repetition_penalty: rp,
                seed: 1,
            },
        };
        let base = gen.generate(&[0], &cfg(1.0)).unwrap();
        let penalized = gen.generate(&[0], &cfg(10.0)).unwrap();
        assert_ne!(
            base, penalized,
            "repetition_penalty should change the sequence"
        );
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
            rope_scaling: None,
            moe: None,
            sliding_window: None,
            activation: Default::default(),
            mla: None,
            ..Default::default()
        };
        let mut r = SplitMix64::new(42);
        let mut vec =
            |n: usize| -> Vec<f32> { (0..n).map(|_| r.next_f32() * 0.1 - 0.05).collect() };
        let layers = vec![LayerTensors {
            q_proj: Weights::from_f32(vec(cfg.q_dim() * hidden)),
            k_proj: Weights::from_f32(vec(cfg.kv_dim() * hidden)),
            v_proj: Weights::from_f32(vec(cfg.kv_dim() * hidden)),
            o_proj: Weights::from_f32(vec(hidden * cfg.q_dim())),
            ffn: Ffn::Dense(ExpertFfn {
                gate: Weights::from_f32(vec(cfg.intermediate_size * hidden)),
                up: Weights::from_f32(vec(cfg.intermediate_size * hidden)),
                down: Weights::from_f32(vec(hidden * cfg.intermediate_size)),
                up_bias: None,
                down_bias: None,
            }),
            input_layernorm: std::vec::from_elem(1.0, hidden),
            post_attention_layernorm: std::vec::from_elem(1.0, hidden),
            ..Default::default()
        }];
        let kernel = CpuKernel::new(cfg, layers).unwrap();
        Generator::new(
            kernel,
            vec(vocab * hidden),
            std::vec::from_elem(1.0, hidden),
            vec(vocab * hidden),
            vocab,
            1e-5,
            KvCacheConfig {
                num_layers: 1,
                num_kv_heads: 2,
                head_dim: 4,
                block_size: 16,
            },
            64,
        )
        .unwrap()
    }

    #[test]
    fn quantized_kv_generation_runs_end_to_end() {
        // Exercises the int8 KV path through the full generate loop (quantized
        // caches, quantizing append, dequantizing attention). Tokens may differ
        // from f32 by argmax flips, so we only require it runs and is well-formed.
        for quant in [crate::forward::KvQuant::Int8, crate::forward::KvQuant::Int4] {
            let gen = attention_generator().with_kv_quant(quant);
            let out = gen
                .generate(
                    &[1, 2, 3],
                    &GenerationConfig {
                        max_new_tokens: 5,
                        eos_token: None,
                        sampler: Sampler::Greedy,
                    },
                )
                .unwrap();
            assert_eq!(out.len(), 5, "{quant:?}");
        }
    }

    #[test]
    fn generate_batch_matches_individual() {
        // Each sequence in a batch must decode identically to running it alone —
        // batching is a throughput optimization, not a semantic change.
        let gen = attention_generator();
        let cfg = GenerationConfig {
            max_new_tokens: 5,
            eos_token: None,
            sampler: Sampler::Greedy,
        };
        let p1: &[u32] = &[1, 2, 3];
        let p2: &[u32] = &[4, 5];
        let batched = gen.generate_batch(&[p1, p2], &cfg).unwrap();
        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0], gen.generate(p1, &cfg).unwrap());
        assert_eq!(batched[1], gen.generate(p2, &cfg).unwrap());
    }

    #[test]
    fn resume_from_snapshot_matches_full_prefill() {
        let gen = attention_generator();
        let prefix = [1u32, 2, 3];
        let full_prompt = [1u32, 2, 3, 4, 5];
        let n = 6;

        // Full prefill of the whole prompt.
        let mut full = gen.start_session(&full_prompt, Sampler::Greedy).unwrap();
        let tokens_full: Vec<u32> = (0..n).map(|_| full.step().unwrap()).collect();

        // Snapshot after the prefix, resume passing the whole prompt (resume
        // derives the suffix from the snapshot position).
        let prefix_sess = gen.start_session(&prefix, Sampler::Greedy).unwrap();
        let snap = prefix_sess.snapshot();
        assert_eq!(snap.position(), prefix.len());
        let mut resumed = gen
            .resume_session(snap, &full_prompt, Sampler::Greedy)
            .unwrap();
        let tokens_resumed: Vec<u32> = (0..n).map(|_| resumed.step().unwrap()).collect();

        // Resuming from the shared prefix is bit-for-bit identical to prefilling
        // the full prompt — the correctness property a prefix cache relies on.
        assert_eq!(tokens_full, tokens_resumed);
    }

    /// With a repetition penalty, a resumed session must match the same request
    /// run without the prefix cache — i.e. the penalty covers the cached *prefix*
    /// tokens, not just the suffix. Before the fix, `seen` held only the suffix,
    /// so a resumed request penalized fewer tokens and drifted.
    #[test]
    fn resume_repetition_penalty_covers_cached_prefix() {
        let gen = attention_generator();
        let prefix = [1u32, 2, 3];
        let full_prompt = [1u32, 2, 3, 4, 5];
        let n = 6;
        let sampler = Sampler::TopPK {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 4.0, // strong, so prefix coverage visibly matters
            seed: 7,
        };

        let mut full = gen.start_session(&full_prompt, sampler).unwrap();
        let tokens_full: Vec<u32> = (0..n).map(|_| full.step().unwrap()).collect();

        let snap = gen.start_session(&prefix, sampler).unwrap().snapshot();
        let mut resumed = gen.resume_session(snap, &full_prompt, sampler).unwrap();
        let tokens_resumed: Vec<u32> = (0..n).map(|_| resumed.step().unwrap()).collect();

        assert_eq!(
            tokens_full, tokens_resumed,
            "resumed penalty must cover the cached prefix, matching a full run"
        );
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

    /// Selecting a prefix before sorting must give the distribution a full sort
    /// gives: the same tokens in the same order, the same probabilities. Covers
    /// a peaked vocabulary (the nucleus fits the first selection) and a flat one
    /// (the selection has to grow).
    #[test]
    fn prefix_selection_matches_a_full_sort() {
        let mut r = SplitMix64::new(11);
        let n = 20_000;
        let peaked: Vec<f32> = (0..n)
            .map(|i| r.next_f32() * 6.0 + if i % 997 == 0 { 12.0 } else { 0.0 })
            .collect();
        let flat: Vec<f32> = (0..n).map(|_| r.next_f32() * 2.0).collect();
        for logits in [&peaked, &flat] {
            for (t, top_p, top_k, min_p) in [
                (0.7, 0.9, 0, 0.0),
                (1.0, 1.0, 0, 0.0),
                (0.8, 0.95, 40, 0.05),
            ] {
                // Reference: sort everything, keep top-k, softmax, then the cuts.
                let mut all: Vec<(f32, u32)> = logits.iter().copied().zip(0u32..).collect();
                all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                if top_k > 0 {
                    all.truncate(top_k);
                }
                let max = all[0].0;
                let w: Vec<f32> = all.iter().map(|&(l, _)| ((l - max) / t).exp()).collect();
                let z: f32 = w.iter().sum();
                let mut want: Vec<(u32, f32)> =
                    all.iter().zip(&w).map(|(&(_, i), &p)| (i, p / z)).collect();
                if top_p < 1.0 {
                    let mut cum = 0.0;
                    let cut = want.iter().position(|&(_, p)| {
                        cum += p;
                        cum >= top_p
                    });
                    want.truncate(cut.map_or(want.len(), |c| c + 1));
                }
                if min_p > 0.0 {
                    let th = min_p * want[0].1;
                    want.truncate(want.iter().take_while(|&&(_, p)| p >= th).count());
                }

                let (idx, probs) = topk_topp_distribution(logits, t, top_p, top_k, min_p);
                let got_ids: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
                let want_ids: Vec<u32> = want.iter().map(|&(i, _)| i).collect();
                assert_eq!(got_ids, want_ids, "t {t} top_p {top_p} top_k {top_k}");
                let wz: f32 = want.iter().map(|&(_, p)| p).sum();
                for (g, &(_, p)) in probs.iter().zip(&want) {
                    assert!((g - p / wz).abs() < 1e-4, "{g} vs {}", p / wz);
                }
            }
        }
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
