//! Paged device KV: one pool of fixed-size blocks per layer, shared by every
//! session.
//!
//! A session used to allocate a full-context K/V buffer per layer on its first
//! token, so each concurrent request cost a whole context's VRAM however short it
//! was. Sessions now take [`KV_BLOCK_ROWS`]-row blocks as they grow and hand them
//! back when they end. The kernels reach a row through the session's block table
//! (`kv_row` in `src/gpu/kernels.cu`).

use crate::error::{DlmError, Result};
use crate::forward::cpu::KvElem;
use crate::gpu::device::DeviceBuffer;
use std::sync::{Arc, Mutex};

/// Values per int8 group ([`KvElem::I8`]): a 4-byte f32 scale, then the codes.
const I8_GROUP: usize = 32;
const I8_GROUP_BYTES: usize = 4 + I8_GROUP;

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
    elem: KvElem,
    kv_dim: usize,
    total_blocks: u32,
    blocks: Mutex<Blocks>,
}

/// Free blocks, and how many holders each block has. A block is shared when a
/// cached prompt prefix and the sessions resumed from it hold it at once.
struct Blocks {
    free: Vec<u32>,
    refs: Vec<u32>,
}

impl KvPool {
    /// A pool of `blocks` blocks for K/V of width `kv_dim`, held as `elem`.
    /// [`KvElem::I8`] needs `kv_dim` to be a whole number of groups.
    pub fn new(blocks: u32, kv_dim: usize, elem: KvElem) -> Result<Self> {
        let elems = blocks as usize * KV_BLOCK_ROWS * kv_dim;
        debug_assert!(elem != KvElem::I8 || kv_dim % I8_GROUP == 0);
        let alloc = || match elem {
            KvElem::F32 => DeviceBuffer::new(elems),
            _ => DeviceBuffer::new_bytes(bytes_of(elem, elems), elems),
        };
        Ok(Self {
            keys: alloc()?,
            values: alloc()?,
            elem,
            kv_dim,
            total_blocks: blocks,
            blocks: Mutex::new(Blocks {
                // Low ids first, so short runs touch the start of the pool.
                free: (0..blocks).rev().collect(),
                refs: vec![0; blocks as usize],
            }),
        })
    }

    pub fn elem(&self) -> KvElem {
        self.elem
    }

    /// Tokens the free blocks can hold.
    pub fn free_tokens(&self) -> usize {
        self.blocks.lock().unwrap().free.len() * KV_BLOCK_ROWS
    }

    pub(crate) fn keys_ptr(&self) -> *mut f32 {
        self.keys.as_mut_ptr()
    }

    pub(crate) fn values_ptr(&self) -> *mut f32 {
        self.values.as_mut_ptr()
    }

    /// Take `n` free blocks, all or none, each with one holder.
    fn take(&self, n: usize) -> Result<Vec<u32>> {
        let mut b = self.blocks.lock().unwrap();
        if b.free.len() < n {
            return Err(DlmError::KvCacheExhausted {
                requested: n as u32,
                free: b.free.len() as u32,
                total: self.total_blocks,
            });
        }
        let at = b.free.len() - n;
        let taken = b.free.split_off(at);
        for &id in &taken {
            b.refs[id as usize] = 1;
        }
        Ok(taken)
    }

    /// Add a holder to each of `ids`.
    fn retain(&self, ids: &[u32]) {
        let mut b = self.blocks.lock().unwrap();
        for &id in ids {
            b.refs[id as usize] += 1;
        }
    }

    /// Drop a holder from each of `ids`, freeing the blocks nobody holds.
    fn give(&self, ids: &[u32]) {
        let mut b = self.blocks.lock().unwrap();
        for &id in ids {
            let r = &mut b.refs[id as usize];
            *r -= 1;
            if *r == 0 {
                b.free.push(id);
            }
        }
    }

    /// Byte offset of the first element of `block` in either pool buffer.
    fn block_offset(&self, block: u32) -> usize {
        bytes_of(self.elem, block as usize * KV_BLOCK_ROWS * self.kv_dim)
    }

    /// Encode f32 rows in the pool's element type. An int8 group takes its
    /// largest magnitude over 127 as its scale, as the kernels' writes do.
    fn encode(&self, rows: &[f32]) -> Vec<u8> {
        match self.elem {
            KvElem::F32 => rows.iter().flat_map(|v| v.to_le_bytes()).collect(),
            KvElem::F16 => rows
                .iter()
                .flat_map(|&v| crate::storage::f32_to_f16(v).to_le_bytes())
                .collect(),
            KvElem::I8 => rows
                .chunks(I8_GROUP)
                .flat_map(|g| {
                    let scale = g.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
                    let codes = g.iter().map(move |&v| {
                        let q = if scale > 0.0 {
                            (v / scale).round()
                        } else {
                            0.0
                        };
                        q as i8 as u8
                    });
                    scale.to_le_bytes().into_iter().chain(codes)
                })
                .collect(),
        }
    }

    /// Decode pool bytes into f32 rows.
    fn decode(&self, bytes: &[u8], out: &mut [f32]) {
        match self.elem {
            KvElem::F32 => {
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            KvElem::F16 => {
                for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *o = crate::storage::f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
            KvElem::I8 => {
                for (o, g) in out
                    .chunks_mut(I8_GROUP)
                    .zip(bytes.chunks_exact(I8_GROUP_BYTES))
                {
                    let scale = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
                    for (v, &c) in o.iter_mut().zip(&g[4..]) {
                        *v = scale * (c as i8) as f32;
                    }
                }
            }
        }
    }
}

/// Bytes `elems` K/V values take as `elem` (a whole number of int8 groups).
fn bytes_of(elem: KvElem, elems: usize) -> usize {
    match elem {
        KvElem::F32 => elems * 4,
        KvElem::F16 => elems * 2,
        KvElem::I8 => elems / I8_GROUP * I8_GROUP_BYTES,
    }
}

/// A session-layer's blocks and its device block table. Dropping it returns the
/// blocks to the pool.
pub struct PagedKv {
    pool: Arc<KvPool>,
    blocks: Vec<u32>,
    /// Rows the table must have room for.
    capacity_rows: usize,
    /// `i32` block ids, room for every block the session may reach. Allocated
    /// on first use, so sharing a prefix (see [`share_prefix`](Self::share_prefix))
    /// cannot fail.
    table: Option<DeviceBuffer>,
    /// Entries of `blocks` already written to `table`.
    table_len: usize,
}

impl PagedKv {
    /// An empty mapping on `pool`, with table room for `capacity_rows` rows.
    pub(crate) fn new(pool: Arc<KvPool>, capacity_rows: usize) -> Result<Self> {
        Ok(Self {
            pool,
            blocks: Vec::new(),
            capacity_rows,
            table: None,
            table_len: 0,
        })
    }

    /// A mapping of the first `rows` rows onto the same blocks, which gain a
    /// holder. `rows` must fall on a block boundary: a partly filled block would
    /// be written by both holders. The new mapping's writes all land past `rows`,
    /// in blocks of its own.
    pub(crate) fn share_prefix(&self, rows: usize) -> Self {
        debug_assert_eq!(rows % KV_BLOCK_ROWS, 0, "shared prefixes are whole blocks");
        let n = (rows / KV_BLOCK_ROWS).min(self.blocks.len());
        let blocks = self.blocks[..n].to_vec();
        self.pool.retain(&blocks);
        Self {
            pool: Arc::clone(&self.pool),
            blocks,
            capacity_rows: self.capacity_rows,
            table: None,
            table_len: 0,
        }
    }

    /// Rows backed by blocks.
    pub(crate) fn rows(&self) -> usize {
        self.blocks.len() * KV_BLOCK_ROWS
    }

    pub(crate) fn pool(&self) -> &KvPool {
        &self.pool
    }

    /// The device block table. Call [`ensure_rows`](Self::ensure_rows) first.
    pub(crate) fn table_ptr(&self) -> *const i32 {
        self.table
            .as_ref()
            .map_or(std::ptr::null(), |t| t.as_ptr() as *const i32)
    }

    /// Make sure rows `0..rows` have blocks, taking new ones from the pool and
    /// recording them in the device table.
    pub(crate) fn ensure_rows(&mut self, rows: usize) -> Result<()> {
        let slots = self.capacity_rows.div_ceil(KV_BLOCK_ROWS).max(1);
        let need = rows.div_ceil(KV_BLOCK_ROWS);
        if need > slots {
            return Err(DlmError::InvalidConfig(format!(
                "GPU KV capacity {} exceeded at position {}",
                slots * KV_BLOCK_ROWS,
                rows - 1
            )));
        }
        if need > self.blocks.len() {
            let new = self.pool.take(need - self.blocks.len())?;
            self.blocks.extend(new);
        }
        if self.table.is_none() {
            self.table = Some(DeviceBuffer::new_bytes(slots * 4, slots)?);
        }
        if self.table_len < self.blocks.len() {
            let bytes: Vec<u8> = self.blocks[self.table_len..]
                .iter()
                .flat_map(|&b| (b as i32).to_le_bytes())
                .collect();
            let table = self.table.as_ref().expect("allocated above");
            table.upload_bytes_at(self.table_len * 4, &bytes)?;
            self.table_len = self.blocks.len();
        }
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
        let mut bytes = Vec::new();
        for (i, (k, v)) in keys
            .chunks_mut(per_block)
            .zip(values.chunks_mut(per_block))
            .enumerate()
        {
            let at = self.pool.block_offset(self.blocks[i]);
            bytes.resize(bytes_of(self.pool.elem, k.len()), 0);
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
            .field("elem", &self.pool.elem)
            .finish()
    }
}

impl Drop for PagedKv {
    fn drop(&mut self) {
        self.pool.give(&self.blocks);
    }
}

/// Every layer's pools for one kernel, created on first use: one per element
/// type a session asks for.
pub struct KvPools {
    layers: Vec<Mutex<[Option<Arc<KvPool>>; 3]>>,
    blocks: u32,
    kv_dim: usize,
    /// int8 groups fit inside heads (`head_dim % 32 == 0`); else int8 asks get
    /// fp16 pools.
    int8: bool,
}

impl KvPools {
    /// Pools for `num_layers` layers of K/V width `kv_dim` (heads of
    /// `head_dim`) holding up to `tokens` tokens each.
    pub fn new(num_layers: usize, kv_dim: usize, head_dim: usize, tokens: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| Mutex::new([None, None, None]))
                .collect(),
            blocks: tokens.div_ceil(KV_BLOCK_ROWS).max(1).min(u32::MAX as usize) as u32,
            kv_dim,
            int8: head_dim > 0 && head_dim % I8_GROUP == 0 && kv_dim % I8_GROUP == 0,
        }
    }

    /// The element type a session asking for `elem` gets: int8 only where its
    /// groups fit inside heads.
    pub fn elem_for(&self, elem: KvElem) -> KvElem {
        if elem == KvElem::I8 && !self.int8 {
            KvElem::F16
        } else {
            elem
        }
    }

    /// Tokens every layer's pool holds.
    pub fn capacity_tokens(&self) -> usize {
        self.blocks as usize * KV_BLOCK_ROWS
    }

    /// Layer `layer`'s pool for `elem` ([`elem_for`](Self::elem_for)),
    /// allocated on first use.
    pub fn get(&self, layer: usize, elem: KvElem) -> Result<Arc<KvPool>> {
        let elem = self.elem_for(elem);
        let mut slots = self.layers[layer].lock().unwrap();
        let slot = &mut slots[elem.code() as usize];
        if let Some(pool) = slot {
            return Ok(Arc::clone(pool));
        }
        let pool = Arc::new(KvPool::new(self.blocks, self.kv_dim, elem)?);
        *slot = Some(Arc::clone(&pool));
        Ok(pool)
    }

    /// Tokens every layer can still take for `elem`. Layers all grow together,
    /// so the fullest layer bounds what a session can still add.
    pub fn free_tokens(&self, elem: KvElem) -> usize {
        let elem = self.elem_for(elem);
        self.layers
            .iter()
            .map(|l| {
                l.lock().unwrap()[elem.code() as usize]
                    .as_ref()
                    .map_or(self.capacity_tokens(), |p| p.free_tokens())
            })
            .min()
            .unwrap_or(0)
    }
}
