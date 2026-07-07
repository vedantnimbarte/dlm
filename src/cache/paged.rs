//! PagedAttention KV cache manager (`specs.md` §2.3).
//!
//! The Key-Value history is the memory that *grows* during generation, and
//! contiguous allocation fragments VRAM badly as sequences start, grow, and
//! finish at different rates. PagedAttention fixes this by splitting the KV
//! history into fixed-size **blocks** (pages) drawn from a shared pool: a
//! sequence holds a *block table* mapping its logical token ranges to whatever
//! physical blocks it was handed, so blocks never need to be contiguous and the
//! pool never fragments.
//!
//! This module manages the *bookkeeping* — block sizing, the free pool, and
//! per-sequence block tables. The physical KV bytes live in the Cache Zone of
//! VRAM (or the host fallback); this layer decides which block holds what and
//! guarantees allocation never silently overflows the budget (returning
//! [`DlmError::KvCacheExhausted`] instead), which is what keeps the engine
//! within its zero-OOM guarantee.

use crate::error::{DlmError, Result};
use crate::model::ModelConfig;
use std::collections::BTreeMap;

/// A physical block index into the KV pool.
pub type BlockId = u32;

/// Bytes per KV element (FP16 cache, per `specs.md` §3.1).
const KV_DTYPE_BYTES: u64 = 2;

/// Geometry of the KV cache: how big one block is, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Transformer layers — KV is stored for every one.
    pub num_layers: u32,
    /// Key/value heads (Grouped-Query Attention shrinks this).
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Tokens covered by a single block (the paging granularity).
    pub block_size: u32,
}

impl KvCacheConfig {
    /// Derive the cache geometry from a model config and a chosen block size
    /// (tokens per page — 16 is a common default).
    pub fn from_model(model: &ModelConfig, block_size: u32) -> Self {
        Self {
            num_layers: model.num_layers,
            num_kv_heads: model.num_kv_heads,
            head_dim: model.head_dim(),
            block_size: block_size.max(1),
        }
    }

    /// Bytes of KV held for a single token across **all** layers
    /// (`2 (K,V) × N_kv_heads × D_head × 2 bytes × N_layers`).
    pub fn bytes_per_token(&self) -> u64 {
        2 * self.num_kv_heads as u64
            * self.head_dim as u64
            * KV_DTYPE_BYTES
            * self.num_layers as u64
    }

    /// Bytes occupied by one physical block (`bytes_per_token × block_size`).
    pub fn bytes_per_block(&self) -> u64 {
        self.bytes_per_token() * self.block_size as u64
    }

    /// Number of blocks needed to hold `tokens` tokens (ceiling division).
    pub fn blocks_for_tokens(&self, tokens: u64) -> u64 {
        tokens.div_ceil(self.block_size as u64)
    }
}

/// Per-sequence state: how many tokens it holds and which physical blocks back
/// them, in logical order.
#[derive(Debug, Clone, Default)]
struct SeqState {
    length: u64,
    blocks: Vec<BlockId>,
}

/// A snapshot of pool utilization for logging/monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheStats {
    pub total_blocks: u32,
    pub used_blocks: u32,
    pub free_blocks: u32,
    pub num_sequences: usize,
    pub used_tokens: u64,
    pub capacity_tokens: u64,
    pub bytes_per_block: u64,
}

impl KvCacheStats {
    /// Fraction of blocks currently allocated (0.0–1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks as f64 / self.total_blocks as f64
        }
    }
}

/// A block-paged KV cache over a fixed pool of physical blocks.
#[derive(Debug)]
pub struct PagedKvCache {
    config: KvCacheConfig,
    total_blocks: u32,
    /// Free physical blocks, used as a LIFO stack.
    free: Vec<BlockId>,
    /// Active sequences and their block tables.
    sequences: BTreeMap<u64, SeqState>,
}

impl PagedKvCache {
    /// Create a cache with an explicit pool size.
    pub fn new(config: KvCacheConfig, total_blocks: u32) -> Self {
        // Hand out low block ids first for predictable, cache-friendly reuse.
        let free: Vec<BlockId> = (0..total_blocks).rev().collect();
        Self {
            config,
            total_blocks,
            free,
            sequences: BTreeMap::new(),
        }
    }

    /// Size the pool to fit within a KV byte budget (e.g. `M_kv_total` from the
    /// [`VramProfiler`](crate::profiler::VramProfiler)).
    pub fn with_budget(config: KvCacheConfig, kv_budget_bytes: u64) -> Self {
        let per_block = config.bytes_per_block().max(1);
        let total_blocks = (kv_budget_bytes / per_block).min(u32::MAX as u64) as u32;
        Self::new(config, total_blocks)
    }

    /// The cache geometry.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Total physical blocks in the pool.
    pub fn total_blocks(&self) -> u32 {
        self.total_blocks
    }

    /// Currently free blocks.
    pub fn free_blocks(&self) -> u32 {
        self.free.len() as u32
    }

    /// Currently allocated blocks.
    pub fn used_blocks(&self) -> u32 {
        self.total_blocks - self.free_blocks()
    }

    /// Total tokens the pool could hold if fully packed.
    pub fn capacity_tokens(&self) -> u64 {
        self.total_blocks as u64 * self.config.block_size as u64
    }

    /// Append `n` tokens to a sequence, creating it if new and allocating blocks
    /// as needed. Atomic: on exhaustion it allocates nothing and returns
    /// [`DlmError::KvCacheExhausted`], leaving the sequence untouched.
    pub fn append_tokens(&mut self, seq_id: u64, n: u64) -> Result<()> {
        let (cur_len, cur_blocks) = self
            .sequences
            .get(&seq_id)
            .map(|s| (s.length, s.blocks.len() as u64))
            .unwrap_or((0, 0));

        let new_len = cur_len + n;
        let needed_blocks = self.config.blocks_for_tokens(new_len);
        let to_alloc = needed_blocks.saturating_sub(cur_blocks);

        if to_alloc > self.free.len() as u64 {
            return Err(DlmError::KvCacheExhausted {
                requested: to_alloc.min(u32::MAX as u64) as u32,
                free: self.free_blocks(),
                total: self.total_blocks,
            });
        }

        // Commit: pull blocks from the pool and extend the sequence's table.
        let seq = self.sequences.entry(seq_id).or_default();
        for _ in 0..to_alloc {
            seq.blocks.push(self.free.pop().expect("checked capacity above"));
        }
        seq.length = new_len;
        Ok(())
    }

    /// Free a sequence, returning all its blocks to the pool. No-op if unknown.
    pub fn free_sequence(&mut self, seq_id: u64) {
        if let Some(state) = self.sequences.remove(&seq_id) {
            self.free.extend(state.blocks);
        }
    }

    /// Number of tokens currently held by a sequence (0 if unknown).
    pub fn sequence_len(&self, seq_id: u64) -> u64 {
        self.sequences.get(&seq_id).map(|s| s.length).unwrap_or(0)
    }

    /// The physical block table backing a sequence, in logical order.
    pub fn block_table(&self, seq_id: u64) -> Option<&[BlockId]> {
        self.sequences.get(&seq_id).map(|s| s.blocks.as_slice())
    }

    /// Number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// A utilization snapshot.
    pub fn stats(&self) -> KvCacheStats {
        KvCacheStats {
            total_blocks: self.total_blocks,
            used_blocks: self.used_blocks(),
            free_blocks: self.free_blocks(),
            num_sequences: self.sequences.len(),
            used_tokens: self.sequences.values().map(|s| s.length).sum(),
            capacity_tokens: self.capacity_tokens(),
            bytes_per_block: self.config.bytes_per_block(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::QuantScheme;

    fn cfg() -> KvCacheConfig {
        // 4 layers, 2 kv-heads, head_dim 8, block of 16 tokens.
        KvCacheConfig {
            num_layers: 4,
            num_kv_heads: 2,
            head_dim: 8,
            block_size: 16,
        }
    }

    #[test]
    fn block_sizing_math() {
        let c = cfg();
        // per token = 2(K,V) * 2 kv-heads * 8 head_dim * 2 bytes * 4 layers = 256
        assert_eq!(c.bytes_per_token(), 256);
        assert_eq!(c.bytes_per_block(), 256 * 16);
        assert_eq!(c.blocks_for_tokens(0), 0);
        assert_eq!(c.blocks_for_tokens(1), 1);
        assert_eq!(c.blocks_for_tokens(16), 1);
        assert_eq!(c.blocks_for_tokens(17), 2);
        assert_eq!(c.blocks_for_tokens(40), 3);
    }

    #[test]
    fn from_model_matches_geometry() {
        let model = ModelConfig::from_json_bytes(
            br#"{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":4,"num_hidden_layers":10,"vocab_size":32000}"#,
            QuantScheme::Fp16,
        )
        .unwrap();
        let c = KvCacheConfig::from_model(&model, 16);
        assert_eq!(c.num_layers, 10);
        assert_eq!(c.num_kv_heads, 4);
        assert_eq!(c.head_dim, 64);
    }

    #[test]
    fn append_allocates_blocks_lazily() {
        let mut cache = PagedKvCache::new(cfg(), 100);
        // 10 tokens → 1 block.
        cache.append_tokens(1, 10).unwrap();
        assert_eq!(cache.block_table(1).unwrap().len(), 1);
        assert_eq!(cache.used_blocks(), 1);
        // 6 more (=16 total) still fits in the first block — no new alloc.
        cache.append_tokens(1, 6).unwrap();
        assert_eq!(cache.block_table(1).unwrap().len(), 1);
        // 1 more (=17) spills into a second block.
        cache.append_tokens(1, 1).unwrap();
        assert_eq!(cache.block_table(1).unwrap().len(), 2);
        assert_eq!(cache.sequence_len(1), 17);
    }

    #[test]
    fn free_returns_blocks_to_pool_without_fragmentation() {
        let mut cache = PagedKvCache::new(cfg(), 4);
        cache.append_tokens(1, 40).unwrap(); // 3 blocks
        assert_eq!(cache.free_blocks(), 1);
        cache.free_sequence(1);
        assert_eq!(cache.free_blocks(), 4);
        assert_eq!(cache.num_sequences(), 0);
        // Reuse the whole pool for a different sequence — no fragmentation.
        cache.append_tokens(2, 64).unwrap(); // exactly 4 blocks
        assert_eq!(cache.used_blocks(), 4);
    }

    #[test]
    fn exhaustion_is_atomic_and_reported() {
        let mut cache = PagedKvCache::new(cfg(), 2); // 32 tokens capacity
        cache.append_tokens(1, 16).unwrap(); // 1 block, 1 free left
        // 40 more → 56 tokens → needs 4 blocks total, have 1, want 3 more, 1 free.
        let err = cache.append_tokens(1, 40).unwrap_err();
        match err {
            DlmError::KvCacheExhausted { requested, free, total } => {
                assert_eq!(requested, 3);
                assert_eq!(free, 1);
                assert_eq!(total, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        // Sequence unchanged by the failed append.
        assert_eq!(cache.sequence_len(1), 16);
        assert_eq!(cache.block_table(1).unwrap().len(), 1);
    }

    #[test]
    fn budget_sizing_and_multi_sequence_sharing() {
        let c = cfg();
        // Budget for exactly 5 blocks.
        let cache_budget = c.bytes_per_block() * 5;
        let mut cache = PagedKvCache::with_budget(c, cache_budget);
        assert_eq!(cache.total_blocks(), 5);

        cache.append_tokens(1, 16).unwrap(); // 1 block
        cache.append_tokens(2, 32).unwrap(); // 2 blocks
        assert_eq!(cache.num_sequences(), 2);
        assert_eq!(cache.used_blocks(), 3);
        assert_eq!(cache.free_blocks(), 2);

        let stats = cache.stats();
        assert_eq!(stats.used_tokens, 48);
        assert_eq!(stats.capacity_tokens, 80);
        assert!((stats.utilization() - 0.6).abs() < 1e-9);
    }
}
