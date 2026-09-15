//! Paged device KV: one pool of fixed-size blocks per layer, shared by every
//! session.
//!
//! A session used to allocate a full-context K/V buffer per layer on its first
//! token, so each concurrent request cost a whole context's VRAM however short it
//! was. Sessions now take [`KV_BLOCK_ROWS`]-row blocks as they grow and hand them
//! back when they end. The kernels reach a row through the session's block table
//! (`kv_row` in `src/gpu/kernels.cu`).

use crate::error::{DlmError, Result};
use crate::gpu::device::DeviceBuffer;
use std::sync::{Arc, Mutex};

/// Rows per block, as a shift: the kernels map rows with shifts, not division.
pub const KV_BLOCK_SHIFT: u32 = 4;
/// Rows (token positions) per block.
pub const KV_BLOCK_ROWS: usize = 1 << KV_BLOCK_SHIFT;
// The scheduler reserves KV in whole blocks of its own constant.
const _: () = assert!(KV_BLOCK_ROWS == crate::batching::KV_BLOCK_TOKENS);

/// One layer's K and V blocks on the device.
pub struct KvPool {
    keys: DeviceBuffer,
    values: DeviceBuffer,
    /// Elements are fp16, not f32.
    half: bool,
    kv_dim: usize,
    total_blocks: u32,
    free: Mutex<Vec<u32>>,
}

impl KvPool {
    /// A pool of `blocks` blocks for K/V of width `kv_dim`, in fp16 when `half`.
    pub fn new(blocks: u32, kv_dim: usize, half: bool) -> Result<Self> {
        let elems = blocks as usize * KV_BLOCK_ROWS * kv_dim;
        let alloc = || {
            if half {
                DeviceBuffer::new_bytes(elems * 2, elems)
            } else {
                DeviceBuffer::new(elems)
            }
        };
        Ok(Self {
            keys: alloc()?,
            values: alloc()?,
            half,
            kv_dim,
            total_blocks: blocks,
            // Low ids first, so short runs touch the start of the pool.
            free: Mutex::new((0..blocks).rev().collect()),
        })
    }

    pub fn half(&self) -> bool {
        self.half
    }

    /// Tokens the free blocks can hold.
    pub fn free_tokens(&self) -> usize {
        self.free.lock().unwrap().len() * KV_BLOCK_ROWS
    }

    pub(crate) fn keys_ptr(&self) -> *mut f32 {
        self.keys.as_mut_ptr()
    }

    pub(crate) fn values_ptr(&self) -> *mut f32 {
        self.values.as_mut_ptr()
    }

    /// Take `n` blocks, all or none.
    fn take(&self, n: usize) -> Result<Vec<u32>> {
        let mut free = self.free.lock().unwrap();
        if free.len() < n {
            return Err(DlmError::KvCacheExhausted {
                requested: n as u32,
                free: free.len() as u32,
                total: self.total_blocks,
            });
        }
        let at = free.len() - n;
        Ok(free.split_off(at))
    }

    fn give(&self, blocks: &[u32]) {
        self.free.lock().unwrap().extend_from_slice(blocks);
    }

    fn elem_bytes(&self) -> usize {
        if self.half {
            2
        } else {
            4
        }
    }

    /// Byte offset of the first element of `block` in either pool buffer.
    fn block_offset(&self, block: u32) -> usize {
        block as usize * KV_BLOCK_ROWS * self.kv_dim * self.elem_bytes()
    }

    /// Encode f32 rows in the pool's element type.
    fn encode(&self, rows: &[f32]) -> Vec<u8> {
        if self.half {
            rows.iter()
                .flat_map(|&v| crate::storage::f32_to_f16(v).to_le_bytes())
                .collect()
        } else {
            rows.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
    }

    /// Decode pool bytes into f32 rows.
    fn decode(&self, bytes: &[u8], out: &mut [f32]) {
        if self.half {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = crate::storage::f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        } else {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
    }
}

/// A session-layer's blocks and its device block table. Dropping it returns the
/// blocks to the pool.
pub struct PagedKv {
    pool: Arc<KvPool>,
    blocks: Vec<u32>,
    /// `i32` block ids, room for every block the session may reach.
    table: DeviceBuffer,
}

impl PagedKv {
    /// An empty mapping on `pool`, with table room for `capacity_rows` rows.
    pub(crate) fn new(pool: Arc<KvPool>, capacity_rows: usize) -> Result<Self> {
        let slots = capacity_rows.div_ceil(KV_BLOCK_ROWS).max(1);
        Ok(Self {
            pool,
            blocks: Vec::new(),
            table: DeviceBuffer::new_bytes(slots * 4, slots)?,
        })
    }

    pub(crate) fn pool(&self) -> &KvPool {
        &self.pool
    }

    pub(crate) fn table_ptr(&self) -> *const i32 {
        self.table.as_ptr() as *const i32
    }

    /// Make sure rows `0..rows` have blocks, taking new ones from the pool and
    /// recording them in the device table.
    pub(crate) fn ensure_rows(&mut self, rows: usize) -> Result<()> {
        let need = rows.div_ceil(KV_BLOCK_ROWS);
        if need <= self.blocks.len() {
            return Ok(());
        }
        if need > self.table.len() {
            return Err(DlmError::InvalidConfig(format!(
                "GPU KV capacity {} exceeded at position {}",
                self.table.len() * KV_BLOCK_ROWS,
                rows - 1
            )));
        }
        let new = self.pool.take(need - self.blocks.len())?;
        let bytes: Vec<u8> = new.iter().flat_map(|&b| (b as i32).to_le_bytes()).collect();
        self.table.upload_bytes_at(self.blocks.len() * 4, &bytes)?;
        self.blocks.extend(new);
        Ok(())
    }

    /// Write f32 rows `0..keys.len() / kv_dim` of keys and values into their
    /// blocks. Rows must already have blocks.
    pub(crate) fn upload_rows(&self, keys: &[f32], values: &[f32]) -> Result<()> {
        let per_block = KV_BLOCK_ROWS * self.pool.kv_dim;
        for (i, (k, v)) in keys
            .chunks(per_block)
            .zip(values.chunks(per_block))
            .enumerate()
        {
            let at = self.pool.block_offset(self.blocks[i]);
            self.pool.keys.upload_bytes_at(at, &self.pool.encode(k))?;
            self.pool.values.upload_bytes_at(at, &self.pool.encode(v))?;
        }
        Ok(())
    }

    /// Read rows `0..keys.len() / kv_dim` of keys and values back as f32.
    pub(crate) fn download_rows(&self, keys: &mut [f32], values: &mut [f32]) -> Result<()> {
        let per_block = KV_BLOCK_ROWS * self.pool.kv_dim;
        let eb = self.pool.elem_bytes();
        let mut bytes = Vec::new();
        for (i, (k, v)) in keys
            .chunks_mut(per_block)
            .zip(values.chunks_mut(per_block))
            .enumerate()
        {
            let at = self.pool.block_offset(self.blocks[i]);
            bytes.resize(k.len() * eb, 0);
            self.pool.keys.download_bytes_at(at, &mut bytes)?;
            self.pool.decode(&bytes, k);
            self.pool.values.download_bytes_at(at, &mut bytes)?;
            self.pool.decode(&bytes, v);
        }
        Ok(())
    }
}

impl std::fmt::Debug for PagedKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedKv")
            .field("blocks", &self.blocks.len())
            .field("half", &self.pool.half)
            .finish()
    }
}

impl Drop for PagedKv {
    fn drop(&mut self) {
        self.pool.give(&self.blocks);
    }
}

/// Every layer's pools for one kernel, created on first use: one per element
/// type a session asks for (f32, fp16).
pub struct KvPools {
    layers: Vec<Mutex<[Option<Arc<KvPool>>; 2]>>,
    blocks: u32,
    kv_dim: usize,
}

impl KvPools {
    /// Pools for `num_layers` layers holding up to `tokens` tokens each.
    pub fn new(num_layers: usize, kv_dim: usize, tokens: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| Mutex::new([None, None])).collect(),
            blocks: tokens.div_ceil(KV_BLOCK_ROWS).max(1).min(u32::MAX as usize) as u32,
            kv_dim,
        }
    }

    /// Tokens every layer's pool holds.
    pub fn capacity_tokens(&self) -> usize {
        self.blocks as usize * KV_BLOCK_ROWS
    }

    /// Layer `layer`'s pool at `half` precision, allocated on first use.
    pub fn get(&self, layer: usize, half: bool) -> Result<Arc<KvPool>> {
        let mut slots = self.layers[layer].lock().unwrap();
        let slot = &mut slots[half as usize];
        if let Some(pool) = slot {
            return Ok(Arc::clone(pool));
        }
        let pool = Arc::new(KvPool::new(self.blocks, self.kv_dim, half)?);
        *slot = Some(Arc::clone(&pool));
        Ok(pool)
    }

    /// Tokens every layer can still take at `half` precision. Layers all grow
    /// together, so the fullest layer bounds what a session can still add.
    pub fn free_tokens(&self, half: bool) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.lock().unwrap()[half as usize]
                    .as_ref()
                    .map_or(self.capacity_tokens(), |p| p.free_tokens())
            })
            .min()
            .unwrap_or(0)
    }
}
