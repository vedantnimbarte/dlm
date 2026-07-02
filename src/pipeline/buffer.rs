//! Streaming-zone buffers for the double-buffered pipeline (`specs.md` §2.2).
//!
//! The Streaming Zone is split into two physical buffers, `A` and `B`. While
//! one is locked and executing on the compute stream, the other is being filled
//! by an asynchronous DMA copy on the memory stream. After each window the roles
//! swap. This module models the buffer identity and a host-backed device buffer
//! so the swap logic is exercisable without a GPU; the CUDA build swaps the
//! backing store for a `cudaMalloc` pointer.

/// Which of the two streaming-zone buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferId {
    A,
    B,
}

impl BufferId {
    /// The opposite buffer — the target of the next prefetch.
    pub fn other(self) -> BufferId {
        match self {
            BufferId::A => BufferId::B,
            BufferId::B => BufferId::A,
        }
    }

    /// The buffer that holds pass `p` under strict A/B ping-pong (even→A, odd→B).
    pub fn for_pass(pass_index: u32) -> BufferId {
        if pass_index % 2 == 0 {
            BufferId::A
        } else {
            BufferId::B
        }
    }
}

/// A fixed-capacity destination for streamed layer weights.
///
/// Host build: backed by a `Vec<u8>` standing in for VRAM. CUDA build (Phase 2
/// GPU path): this becomes a thin wrapper over a `cudaMalloc` device pointer and
/// the copy target of `cudaMemcpyAsync`.
#[derive(Debug)]
pub struct DeviceBuffer {
    storage: Vec<u8>,
    /// Bytes currently valid (written by the last copy).
    filled: usize,
}

impl DeviceBuffer {
    /// Allocate a buffer able to hold `capacity` bytes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            storage: vec![0u8; capacity],
            filled: 0,
        }
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// Bytes written by the most recent copy.
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Copy `src` into this buffer (the host-fallback analogue of a
    /// `cudaMemcpyAsync` host→device transfer). Returns the number of bytes
    /// written, or an error if `src` exceeds capacity.
    pub fn copy_in(&mut self, src: &[u8]) -> crate::error::Result<usize> {
        if src.len() > self.storage.len() {
            return Err(crate::error::FlipError::HostAlloc {
                bytes: src.len(),
                align: self.storage.len(),
            });
        }
        self.storage[..src.len()].copy_from_slice(src);
        self.filled = src.len();
        Ok(src.len())
    }

    /// View of the currently valid bytes (what a compute kernel would read).
    pub fn as_slice(&self) -> &[u8] {
        &self.storage[..self.filled]
    }
}
