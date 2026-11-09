//! A model split between two kernels: the first `k` layers on one, the rest on
//! another.
//!
//! The case it exists for is a model that does not fit in VRAM. Streaming moves
//! every non-resident layer across PCIe on every token; on a card in a PCIe 3.0
//! x16 slot that is 12 GB/s at best and 6-8 in practice, through a pinned copy.
//! The same layers already sit in host RAM, which a desktop's two DDR4 channels
//! read at 20-40 GB/s. So running them on the CPU, where they are, can beat
//! moving them -- the split llama.cpp makes with `-ngl`. Only the residual
//! (`hidden_size` floats) crosses at the boundary, once per token.
//!
//! The front kernel holds model layers `0..k` under their own indices; the back
//! kernel holds `k..` and is told where it starts (`CpuKernel::for_layers`), so
//! each layer's config is looked up by its index in the model either way. Each
//! layer's KV history lives with the kernel that runs it.

use crate::error::Result;
use crate::forward::cpu::KvLayerCache;
use crate::forward::kernel::ComputeKernel;

/// Layers `0..front.num_layers()` on `front`, the rest on `back`.
pub struct SplitKernel<F, B> {
    front: F,
    back: B,
}

impl<F: ComputeKernel, B: ComputeKernel> SplitKernel<F, B> {
    /// `back` must number its layers from 0 for model layer
    /// `front.num_layers()`.
    pub fn new(front: F, back: B) -> Self {
        Self { front, back }
    }

    fn split(&self) -> u32 {
        self.front.num_layers()
    }
}

impl<F: ComputeKernel, B: ComputeKernel> ComputeKernel for SplitKernel<F, B> {
    fn num_layers(&self) -> u32 {
        self.front.num_layers() + self.back.num_layers()
    }

    fn hidden_size(&self) -> usize {
        self.front.hidden_size()
    }

    fn kv_dim(&self) -> usize {
        self.front.kv_dim()
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()> {
        let k = self.split();
        if layer < k {
            self.front.run_block(layer, hidden, kv, position)
        } else {
            self.back.run_block(layer - k, hidden, kv, position)
        }
    }

    fn run_block_batched(
        &self,
        layer: u32,
        hiddens: &mut [&mut [f32]],
        kvs: &mut [&mut KvLayerCache],
        positions: &[usize],
    ) -> Result<()> {
        let k = self.split();
        if layer < k {
            self.front.run_block_batched(layer, hiddens, kvs, positions)
        } else {
            self.back
                .run_block_batched(layer - k, hiddens, kvs, positions)
        }
    }

    /// The whole batch through the front's layers, then through the back's, so
    /// each kernel keeps its own batching (the GPU's one fused call per layer).
    fn decode_batch(
        &self,
        hiddens: &mut [&mut [f32]],
        kv_layers: &mut [&mut [KvLayerCache]],
        positions: &[usize],
    ) -> Result<()> {
        let k = self.split() as usize;
        let (mut fronts, mut backs): (Vec<&mut [KvLayerCache]>, Vec<&mut [KvLayerCache]>) =
            kv_layers.iter_mut().map(|kv| kv.split_at_mut(k)).unzip();
        self.front.decode_batch(hiddens, &mut fronts, positions)?;
        self.back.decode_batch(hiddens, &mut backs, positions)
    }

    fn prefill(
        &self,
        hiddens: &mut [f32],
        kv_layers: &mut [KvLayerCache],
        start: usize,
    ) -> Result<()> {
        let (front, back) = kv_layers.split_at_mut(self.split() as usize);
        self.front.prefill(hiddens, front, start)?;
        self.back.prefill(hiddens, back, start)
    }

    fn kv_free_tokens(&self, fp16: bool) -> Option<usize> {
        match (
            self.front.kv_free_tokens(fp16),
            self.back.kv_free_tokens(fp16),
        ) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn set_kv_pool_tokens(&mut self, tokens: usize) {
        self.front.set_kv_pool_tokens(tokens);
        self.back.set_kv_pool_tokens(tokens);
    }

    fn stream_stats(&self) -> Option<crate::forward::streaming::StreamStats> {
        self.front
            .stream_stats()
            .or_else(|| self.back.stream_stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvCacheConfig, PagedKvCache};
    use crate::forward::{BlockConfig, CpuKernel, ForwardOrchestrator, KvQuant, LayerTensors};
    use crate::forward::{ExpertFfn, Ffn, Weights};

    fn layers(cfg: &BlockConfig, n: u32) -> Vec<LayerTensors> {
        let mut seed = 0x5917u64;
        let mut v = |len: usize| {
            Weights::from_f32(
                (0..len)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
                    })
                    .collect(),
            )
        };
        (0..n)
            .map(|_| LayerTensors {
                q_proj: v(cfg.q_dim() * cfg.hidden_size),
                k_proj: v(cfg.kv_dim() * cfg.hidden_size),
                v_proj: v(cfg.kv_dim() * cfg.hidden_size),
                o_proj: v(cfg.hidden_size * cfg.q_dim()),
                ffn: Ffn::Dense(ExpertFfn {
                    gate: v(cfg.intermediate_size * cfg.hidden_size),
                    up: v(cfg.intermediate_size * cfg.hidden_size),
                    down: v(cfg.hidden_size * cfg.intermediate_size),
                    up_bias: None,
                    down_bias: None,
                }),
                input_layernorm: vec![1.0; cfg.hidden_size],
                post_attention_layernorm: vec![1.0; cfg.hidden_size],
                ..Default::default()
            })
            .collect()
    }

    /// A split model computes exactly what the whole one does: the same blocks,
    /// in the same order, through prefill and decode.
    #[test]
    fn split_matches_whole() {
        let cfg = BlockConfig {
            hidden_size: 16,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            sliding_window: Some(2),
            ..Default::default()
        };
        let all = layers(&cfg, 4);
        let kv = KvCacheConfig {
            num_layers: 4,
            num_kv_heads: 1,
            head_dim: 8,
            block_size: 16,
        };
        let whole = CpuKernel::new(cfg, all.clone()).unwrap();
        let split = SplitKernel::new(
            CpuKernel::new(cfg, all[..1].to_vec()).unwrap(),
            CpuKernel::for_layers(cfg, all[1..].to_vec(), 1).unwrap(),
        );
        let mut a = ForwardOrchestrator::new(whole, PagedKvCache::new(kv, 4), KvQuant::None);
        let mut b = ForwardOrchestrator::new(split, PagedKvCache::new(kv, 4), KvQuant::None);
        let mut prompt: Vec<f32> = (0..3 * 16).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let mut prompt_b = prompt.clone();
        a.prefill(&mut prompt).unwrap();
        b.prefill(&mut prompt_b).unwrap();
        assert_eq!(prompt, prompt_b);
        for step in 0..4 {
            let mut h: Vec<f32> = (0..16).map(|i| (i + step) as f32 * 0.05).collect();
            let mut g = h.clone();
            a.decode_token(&mut h).unwrap();
            b.decode_token(&mut g).unwrap();
            assert_eq!(h, g, "step {step}");
        }
    }
}
