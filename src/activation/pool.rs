//! Residual activation pool (`specs.md` §2.3).
//!
//! Layer streaming frees a block's weights the moment its compute finishes — but
//! the **residual stream** (the running hidden state that each transformer block
//! adds into) must survive to feed the next block. Alongside it, every block
//! needs scratch space for intermediate activations. Allocating and freeing
//! those arrays 80× per token would churn the allocator and fragment memory.
//!
//! The [`ActivationPool`] recycles fixed-size `f32` buffers: a buffer is
//! acquired for a layer's work and released back into a free list when done, so
//! the whole forward pass reuses a handful of buffers instead of allocating one
//! per layer. The pool is bounded — acquiring beyond its cap returns
//! [`DlmError::ActivationPoolExhausted`] rather than growing without limit.

use crate::error::{DlmError, Result};

/// Bytes per `f32` element.
const F32_BYTES: usize = 4;

/// A checked-out activation buffer of `buffer_elems` `f32`s. Returned to its
/// pool via [`ActivationPool::release`] (or automatically by
/// [`ActivationPool::with_buffer`]).
#[derive(Debug)]
pub struct ActivationBuffer {
    data: Vec<f32>,
}

impl ActivationBuffer {
    /// Element count.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable view.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable view — where a layer writes its output activations.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Accumulate a skip connection: `self += other`, elementwise. This is the
    /// residual add `h = h + f(h)` that carries state across streamed layers.
    pub fn add_assign_slice(&mut self, other: &[f32]) -> Result<()> {
        if other.len() != self.data.len() {
            return Err(DlmError::ShapeMismatch {
                expected: self.data.len(),
                got: other.len(),
            });
        }
        for (dst, &src) in self.data.iter_mut().zip(other) {
            *dst += src;
        }
        Ok(())
    }
}

/// Utilization snapshot for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationPoolStats {
    pub buffer_elems: usize,
    pub max_buffers: usize,
    pub in_use: usize,
    pub pooled: usize,
    pub allocations: u64,
    pub reuses: u64,
    pub peak_in_use: usize,
    pub bytes_per_buffer: usize,
}

impl ActivationPoolStats {
    /// Fraction of acquisitions served by reuse rather than fresh allocation.
    pub fn reuse_rate(&self) -> f64 {
        let total = self.allocations + self.reuses;
        if total == 0 {
            0.0
        } else {
            self.reuses as f64 / total as f64
        }
    }
}

/// A bounded pool of reusable fixed-size activation buffers.
#[derive(Debug)]
pub struct ActivationPool {
    buffer_elems: usize,
    max_buffers: usize,
    free: Vec<Vec<f32>>,
    in_use: usize,
    allocations: u64,
    reuses: u64,
    peak_in_use: usize,
}

impl ActivationPool {
    /// A pool of at most `max_buffers` buffers, each `buffer_elems` `f32`s.
    pub fn new(buffer_elems: usize, max_buffers: usize) -> Self {
        Self {
            buffer_elems: buffer_elems.max(1),
            max_buffers: max_buffers.max(1),
            free: Vec::new(),
            in_use: 0,
            allocations: 0,
            reuses: 0,
            peak_in_use: 0,
        }
    }

    /// Size the pool from a byte budget: as many `buffer_elems`-sized buffers as
    /// fit in `max_bytes` (at least one).
    pub fn with_byte_budget(buffer_elems: usize, max_bytes: u64) -> Self {
        let elems = buffer_elems.max(1);
        let per_buffer = (elems * F32_BYTES) as u64;
        let max_buffers = (max_bytes / per_buffer).max(1).min(usize::MAX as u64) as usize;
        Self::new(elems, max_buffers)
    }

    /// Elements per buffer.
    pub fn buffer_elems(&self) -> usize {
        self.buffer_elems
    }

    /// Buffer cap.
    pub fn max_buffers(&self) -> usize {
        self.max_buffers
    }

    /// Currently checked-out buffers.
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// Acquire a zero-initialized buffer, reusing a pooled one when available.
    /// Errors with [`DlmError::ActivationPoolExhausted`] when all buffers are
    /// checked out and the cap is reached.
    pub fn acquire(&mut self) -> Result<ActivationBuffer> {
        let data = if let Some(mut buf) = self.free.pop() {
            buf.iter_mut().for_each(|x| *x = 0.0);
            self.reuses += 1;
            buf
        } else if self.in_use < self.max_buffers {
            self.allocations += 1;
            vec![0.0f32; self.buffer_elems]
        } else {
            return Err(DlmError::ActivationPoolExhausted {
                in_use: self.in_use,
                max: self.max_buffers,
            });
        };

        self.in_use += 1;
        self.peak_in_use = self.peak_in_use.max(self.in_use);
        Ok(ActivationBuffer { data })
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, buffer: ActivationBuffer) {
        debug_assert_eq!(buffer.data.len(), self.buffer_elems);
        self.free.push(buffer.data);
        self.in_use = self.in_use.saturating_sub(1);
    }

    /// Acquire a buffer, run `f` over it, and release it automatically.
    pub fn with_buffer<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut [f32]) -> R,
    {
        let mut buffer = self.acquire()?;
        let result = f(buffer.as_mut_slice());
        self.release(buffer);
        Ok(result)
    }

    /// A utilization snapshot.
    pub fn stats(&self) -> ActivationPoolStats {
        ActivationPoolStats {
            buffer_elems: self.buffer_elems,
            max_buffers: self.max_buffers,
            in_use: self.in_use,
            pooled: self.free.len(),
            allocations: self.allocations,
            reuses: self.reuses,
            peak_in_use: self.peak_in_use,
            bytes_per_buffer: self.buffer_elems * F32_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_released_buffers() {
        let mut pool = ActivationPool::new(4, 8);
        // Acquire and release repeatedly: exactly one real allocation.
        for _ in 0..10 {
            let buf = pool.acquire().unwrap();
            pool.release(buf);
        }
        let s = pool.stats();
        assert_eq!(s.allocations, 1);
        assert_eq!(s.reuses, 9);
        assert_eq!(s.in_use, 0);
        assert_eq!(s.pooled, 1);
        assert_eq!(s.peak_in_use, 1);
    }

    #[test]
    fn concurrent_buffers_allocate_up_to_cap() {
        let mut pool = ActivationPool::new(4, 3);
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        let c = pool.acquire().unwrap();
        assert_eq!(pool.in_use(), 3);
        // Fourth concurrent acquire exceeds the cap.
        assert!(pool.acquire().is_err());
        pool.release(a);
        // Now one is free again.
        let d = pool.acquire().unwrap();
        assert_eq!(pool.in_use(), 3);
        pool.release(b);
        pool.release(c);
        pool.release(d);
        assert_eq!(pool.stats().allocations, 3);
    }

    #[test]
    fn acquired_buffer_is_zeroed_on_reuse() {
        let mut pool = ActivationPool::new(3, 2);
        let mut buf = pool.acquire().unwrap();
        buf.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0]);
        pool.release(buf);
        // Reacquired buffer must come back zeroed.
        let buf2 = pool.acquire().unwrap();
        assert_eq!(buf2.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn residual_add_accumulates() {
        let mut pool = ActivationPool::new(3, 2);
        let mut h = pool.acquire().unwrap();
        h.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0]);
        // h = h + f(h)
        h.add_assign_slice(&[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(h.as_slice(), &[11.0, 22.0, 33.0]);
        // Mismatched length rejected.
        assert!(h.add_assign_slice(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn with_buffer_auto_releases() {
        let mut pool = ActivationPool::new(2, 1);
        let sum = pool
            .with_buffer(|b| {
                b.copy_from_slice(&[2.0, 5.0]);
                b.iter().sum::<f32>()
            })
            .unwrap();
        assert_eq!(sum, 7.0);
        assert_eq!(pool.in_use(), 0);
        // The single buffer was returned and is reusable.
        assert!(pool.with_buffer(|_| ()).is_ok());
        assert_eq!(pool.stats().allocations, 1);
    }

    #[test]
    fn byte_budget_sizes_pool() {
        // 4 elems * 4 bytes = 16 bytes/buffer; 64-byte budget → 4 buffers.
        let pool = ActivationPool::with_byte_budget(4, 64);
        assert_eq!(pool.max_buffers(), 4);
        assert_eq!(pool.stats().bytes_per_buffer, 16);
    }
}
