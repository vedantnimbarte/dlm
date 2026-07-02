//! Cache Zone: PagedAttention KV cache + tiered CPU-RAM layer cache.

pub mod paged;
pub mod ram;

pub use paged::{BlockId, KvCacheConfig, KvCacheStats, PagedKvCache};
pub use ram::{LayerRamCache, RamCacheStats};
